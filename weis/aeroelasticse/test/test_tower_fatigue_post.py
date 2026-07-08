"""
Deterministic regression test for TowerFatiguePostFrame.

This test does not require QBlade, OpenFAST, or any solver license: it builds
synthetic tower-load time-series ``.p`` files directly (the same file format
QBlade/pCrunch write) and a synthetic tower geometry, runs the component, and
compares the resulting fatigue damage against stored golden values.

Purpose: guard the numerical behavior of TowerFatiguePostFrame against silent
regressions during refactors (comment fixes, caching, ponytail cleanup, mean
stress correction, etc). If a code change is *intended* to change the fatigue
numbers, regenerate the golden values with:

    python weis/aeroelasticse/test/test_tower_fatigue_post.py --train
"""

import os
import sys
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import openmdao.api as om
import unittest

from weis.aeroelasticse import tower_fatigue_post as tftp
from weis.aeroelasticse.tower_fatigue_post import (
    TowerFatiguePostFrame,
    _smoke_test_scale_to_si,
)
from weis.test.utils import compare_regression_values

this_dir = os.path.dirname(os.path.realpath(__file__))
truth_file = os.path.join(this_dir, "tower_fatigue_regression_values.pkl")

N_FULL = 4
N_THETA = 8
GRID_POSITIONS = [0.0, 0.33, 0.67, 1.0]


def _build_synthetic_tower_fatigue_case(tmpdir, probabilities=None):
    """Build a synthetic tower geometry plus two synthetic load-time-series
    cases on disk. Returns (geometry_dict, discrete_inputs_dict).

    The signals are fully deterministic (no randomness) so the test is
    reproducible across machines and Python versions.
    """
    z_full = np.array([0.0, 30.0, 60.0, 90.0])
    outer_diameter_full = np.array([8.0, 7.0, 6.0, 5.0])
    t_full = np.array([0.03, 0.025, 0.02])

    time = np.arange(0.0, 60.0, 0.1)

    case_files = []
    # (amplitude, frequency, phase, probability) — deliberately distinct
    # per case so both contribute differently to lifetime damage.
    case_defs = [(1.0, 0.3, 0.0, 0.6), (1.5, 0.5, 0.7, 0.4)]
    if probabilities is not None:
        case_defs = [
            (amp, freq, phase, prob)
            for (amp, freq, phase, _prob), prob in zip(case_defs, probabilities)
        ]

    for i_case, (amp, freq, phase, _prob) in enumerate(case_defs):
        cols = {"Time": time}
        for ig, pos in enumerate(GRID_POSITIONS):
            base_load = 5.0e5 * (1.0 - pos)  # larger loads near tower base
            cols[f"Fz_g{ig}"] = base_load + amp * 2.0e4 * np.sin(freq * time + phase + ig)
            cols[f"Mx_g{ig}"] = amp * 3.0e6 * (1.0 - pos) * np.sin(freq * time * 1.3 + phase)
            cols[f"My_g{ig}"] = amp * 2.5e6 * (1.0 - pos) * np.cos(freq * time * 0.9 + phase)
        df = pd.DataFrame(cols)
        fname = f"case_{i_case}.p"
        df.to_pickle(os.path.join(tmpdir, fname))
        case_files.append(fname)

    load_channels = []
    for ig, pos in enumerate(GRID_POSITIONS):
        load_channels.append({
            "twr_sec_pos": pos,
            "keys": {"Fz": f"Fz_g{ig}", "Mx": f"Mx_g{ig}", "My": f"My_g{ig}"},
            "scale_to_si": {"Fz": 1.0, "Mx": 1.0, "My": 1.0},
        })

    geometry = {
        "z_full": z_full,
        "outer_diameter_full": outer_diameter_full,
        "t_full": t_full,
    }

    discrete = {
        "tower_fatigue_ts_dir": tmpdir,
        "tower_fatigue_case_names": tuple(f"case_{i}" for i in range(len(case_defs))),
        "tower_fatigue_case_probability": tuple(p for *_r, p in case_defs),
        "tower_fatigue_case_files": tuple(case_files),
        "tower_fatigue_load_channels": tuple(load_channels),
    }

    return geometry, discrete


def _run_tower_fatigue_component(
    sn_model="bilinear",
    n_workers=None,
    probabilities=None,
    mutate_discrete=None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        geometry, discrete = _build_synthetic_tower_fatigue_case(
            tmpdir,
            probabilities=probabilities,
        )
        if mutate_discrete is not None:
            mutate_discrete(discrete)

        prob = om.Problem()
        component_options = {
            "n_full": N_FULL,
            "n_theta": N_THETA,
            "sn_model": sn_model,
        }
        if n_workers is not None:
            component_options["modeling_options"] = {
                "TowerFatigue": {"n_workers": n_workers}
            }
        prob.model.add_subsystem(
            "tower_fatigue_post",
            TowerFatiguePostFrame(**component_options),
        )
        prob.setup()

        for key, val in geometry.items():
            prob.set_val(f"tower_fatigue_post.{key}", val)
        for key, val in discrete.items():
            prob.set_val(f"tower_fatigue_post.{key}", val)

        prob.run_model()

        return {
            "fatigue_damage": prob.get_val("tower_fatigue_post.fatigue_damage").copy(),
            "constr_fatigue": prob.get_val("tower_fatigue_post.constr_fatigue").copy(),
            "section_A": prob.get_val("tower_fatigue_post.section_A").copy(),
        }


class TestTowerFatiguePost(unittest.TestCase):

    def assert_outputs_allclose(self, actual, expected):
        for key in ("fatigue_damage", "constr_fatigue", "section_A"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=0.0)

    def test_regression_bilinear(self):
        values = _run_tower_fatigue_component(sn_model="bilinear")
        compare_regression_values(
            [values],
            "tower_fatigue_regression_values.pkl",
            directory=this_dir,
            tol=1e-6,
        )

    def test_damage_is_nonzero_and_decreases_up_the_tower(self):
        # Physical sanity check independent of the golden file: with larger
        # loads concentrated at the tower base, damage should be highest
        # there and should not be degenerate (zero/NaN/inf) anywhere.
        values = _run_tower_fatigue_component(sn_model="bilinear")
        damage = values["fatigue_damage"]
        self.assertTrue(np.all(np.isfinite(damage)))
        self.assertTrue(np.all(damage > 0.0))
        self.assertTrue(np.all(np.diff(damage) <= 0.0))

    def test_scale_to_si_smoke_test(self):
        # Fold the module's built-in unit-conversion smoke test into the
        # pytest-discovered suite so CI actually exercises it.
        _smoke_test_scale_to_si()

    def test_default_serial_matches_explicit_serial_worker_path(self):
        default_values = _run_tower_fatigue_component(sn_model="bilinear")
        serial_values = _run_tower_fatigue_component(
            sn_model="bilinear",
            n_workers=1,
        )

        self.assert_outputs_allclose(serial_values, default_values)

    def test_parallel_workers_match_serial(self):
        serial_values = _run_tower_fatigue_component(
            sn_model="bilinear",
            n_workers=1,
        )

        with mock.patch.object(tftp.os, "cpu_count", return_value=2):
            parallel_values = _run_tower_fatigue_component(
                sn_model="bilinear",
                n_workers=2,
            )

        self.assert_outputs_allclose(parallel_values, serial_values)

    def test_zero_probability_case_is_not_loaded(self):
        def point_inactive_case_to_missing_file(discrete):
            case_files = list(discrete["tower_fatigue_case_files"])
            case_files[1] = "missing_zero_probability_case.p"
            discrete["tower_fatigue_case_files"] = tuple(case_files)

        values = _run_tower_fatigue_component(
            sn_model="bilinear",
            probabilities=(1.0, 0.0),
            mutate_discrete=point_inactive_case_to_missing_file,
        )

        self.assertTrue(np.all(np.isfinite(values["fatigue_damage"])))

    def test_one_active_case_uses_serial_path_when_many_workers_requested(self):
        with mock.patch.object(tftp.os, "cpu_count", return_value=4):
            with mock.patch.object(
                tftp.concurrent.futures,
                "ProcessPoolExecutor",
            ) as executor_cls:
                _run_tower_fatigue_component(
                    sn_model="bilinear",
                    n_workers=4,
                    probabilities=(1.0, 0.0),
                )

        executor_cls.assert_not_called()

    def test_parallel_collection_preserves_case_index_with_out_of_order_futures(self):
        active_case_requests = [
            {"case_index": 0, "case_name": "case_0", "case_file": "case_0.p"},
            {"case_index": 1, "case_name": "case_1", "case_file": "case_1.p"},
        ]
        shared_payload = {}

        def fake_worker(case_request, _shared_payload):
            value = 1.0 if case_request["case_index"] == 0 else 10.0
            return case_request["case_index"], np.full((2, 2), value)

        class FakeExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, case_request, payload):
                future = tftp.concurrent.futures.Future()
                future.set_result(fn(case_request, payload))
                return future

        with mock.patch.object(tftp.os, "cpu_count", return_value=2):
            with mock.patch.object(tftp, "_process_tower_fatigue_case", side_effect=fake_worker):
                with mock.patch.object(
                    tftp.concurrent.futures,
                    "ProcessPoolExecutor",
                    FakeExecutor,
                ):
                    with mock.patch.object(
                        tftp.concurrent.futures,
                        "as_completed",
                        side_effect=lambda futures: list(reversed(list(futures))),
                    ):
                        result_by_case, effective_n_workers = (
                            tftp._run_tower_fatigue_case_workers(
                                active_case_requests=active_case_requests,
                                shared_payload=shared_payload,
                                requested_n_workers=2,
                            )
                        )

        self.assertEqual(effective_n_workers, 2)
        damage_theta = np.zeros((2, 2))
        for case_request in active_case_requests:
            damage_theta += result_by_case[case_request["case_index"]]

        np.testing.assert_allclose(damage_theta, np.full((2, 2), 11.0))

    def test_worker_error_message_includes_case_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_request = {
                "case_index": 7,
                "case_name": "bad_case",
                "case_file": "missing_case.p",
                "case_probability": 1.0,
            }
            shared_payload = {
                "ts_dir": tmpdir,
                "tower_grid": np.array([0.0, 1.0]),
                "load_key_map": [{"Fz": "Fz0", "Mx": "Mx0", "My": "My0"}],
                "load_scale_map": [{"Fz": 1.0, "Mx": 1.0, "My": 1.0}],
            }

            with self.assertRaises(RuntimeError) as ctx:
                tftp._process_tower_fatigue_case(case_request, shared_payload)

        message = str(ctx.exception)
        self.assertIn("case_index=7", message)
        self.assertIn("case_name='bad_case'", message)
        self.assertIn("case_file='missing_case.p'", message)
        self.assertIsNotNone(ctx.exception.__cause__)

    def test_parquet_loading_uses_filtered_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            case_file = "case.parquet"
            open(os.path.join(tmpdir, case_file), "wb").close()
            time = np.array([0.0, 1.0, 2.0])
            data = pd.DataFrame({
                "Time": time,
                "Fz0": np.array([1.0, 2.0, 3.0]),
                "Mx0": np.array([4.0, 5.0, 6.0]),
                "My0": np.array([7.0, 8.0, 9.0]),
                "Fz1": np.array([2.0, 3.0, 4.0]),
                "Mx1": np.array([5.0, 6.0, 7.0]),
                "My1": np.array([8.0, 9.0, 10.0]),
                "unused": np.array([100.0, 101.0, 102.0]),
            })
            seen_columns = []

            def fake_read_parquet(_path, columns=None):
                seen_columns.append(columns)
                return data.loc[:, columns]

            with mock.patch.object(tftp.pd, "read_parquet", side_effect=fake_read_parquet):
                time_out, Fz_grid, Mx_grid, My_grid = (
                    tftp._load_case_tower_loads_on_solver_grid(
                        ts_dir=tmpdir,
                        case_file=case_file,
                        tower_grid=np.array([0.0, 1.0]),
                        load_key_map=[
                            {"Fz": "Fz0", "Mx": "Mx0", "My": "My0"},
                            {"Fz": "Fz1", "Mx": "Mx1", "My": "My1"},
                        ],
                        load_scale_map=[
                            {"Fz": 1.0, "Mx": 1.0, "My": 1.0},
                            {"Fz": 1.0, "Mx": 1.0, "My": 1.0},
                        ],
                    )
                )

        self.assertEqual(
            seen_columns[0],
            ["Time", "Fz0", "Mx0", "My0", "Fz1", "Mx1", "My1"],
        )
        np.testing.assert_allclose(time_out, time)
        np.testing.assert_allclose(Fz_grid[0, :], data["Fz0"].to_numpy())
        np.testing.assert_allclose(Mx_grid[1, :], data["Mx1"].to_numpy())
        np.testing.assert_allclose(My_grid[1, :], data["My1"].to_numpy())


if __name__ == "__main__":
    if "--train" in sys.argv:
        values = _run_tower_fatigue_component(sn_model="bilinear")
        compare_regression_values(
            [values],
            "tower_fatigue_regression_values.pkl",
            directory=this_dir,
            train=True,
        )
        print(f"Wrote golden values to {truth_file}")
    else:
        unittest.main()
