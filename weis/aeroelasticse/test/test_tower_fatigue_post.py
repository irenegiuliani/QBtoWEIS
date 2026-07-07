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
import contextlib
import io

import numpy as np
import pandas as pd
import openmdao.api as om
import unittest

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


def _build_synthetic_tower_fatigue_case(tmpdir, include_zero_probability_case=False):
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

    if include_zero_probability_case:
        discrete["tower_fatigue_case_names"] = (
            discrete["tower_fatigue_case_names"] + ("case_zero_probability",)
        )
        discrete["tower_fatigue_case_probability"] = (
            discrete["tower_fatigue_case_probability"] + (0.0,)
        )
        discrete["tower_fatigue_case_files"] = (
            discrete["tower_fatigue_case_files"] + ("missing_zero_probability_case.p",)
        )

    return geometry, discrete


def _run_tower_fatigue_component(
    sn_model="bilinear",
    profile=False,
    include_zero_probability_case=False,
    capture_stdout=False,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        geometry, discrete = _build_synthetic_tower_fatigue_case(
            tmpdir,
            include_zero_probability_case=include_zero_probability_case,
        )

        prob = om.Problem()
        prob.model.add_subsystem(
            "tower_fatigue_post",
            TowerFatiguePostFrame(
                n_full=N_FULL,
                n_theta=N_THETA,
                sn_model=sn_model,
                modeling_options={"TowerFatigue": {"profile": profile}},
            ),
        )
        prob.setup()

        for key, val in geometry.items():
            prob.set_val(f"tower_fatigue_post.{key}", val)
        for key, val in discrete.items():
            prob.set_val(f"tower_fatigue_post.{key}", val)

        stdout = io.StringIO()
        if capture_stdout:
            with contextlib.redirect_stdout(stdout):
                prob.run_model()
        else:
            prob.run_model()

        values = {
            "fatigue_damage": prob.get_val("tower_fatigue_post.fatigue_damage").copy(),
            "constr_fatigue": prob.get_val("tower_fatigue_post.constr_fatigue").copy(),
            "section_A": prob.get_val("tower_fatigue_post.section_A").copy(),
        }

        if capture_stdout:
            values["stdout"] = stdout.getvalue()

        return values


class TestTowerFatiguePost(unittest.TestCase):

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

    def test_profile_does_not_change_results_and_skips_zero_probability_cases(self):
        values_no_profile = _run_tower_fatigue_component(
            sn_model="bilinear",
            profile=False,
            include_zero_probability_case=True,
            capture_stdout=True,
        )
        values_profile = _run_tower_fatigue_component(
            sn_model="bilinear",
            profile=True,
            include_zero_probability_case=True,
            capture_stdout=True,
        )

        for key in ("fatigue_damage", "constr_fatigue", "section_A"):
            np.testing.assert_allclose(
                values_profile[key],
                values_no_profile[key],
                rtol=0.0,
                atol=0.0,
            )

        self.assertNotIn("[TowerFatigue]", values_no_profile["stdout"])
        self.assertIn("[TowerFatigue][SUMMARY]", values_profile["stdout"])
        self.assertIn("skipped_zero_probability_cases = 1", values_profile["stdout"])


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
