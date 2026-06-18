import importlib.util
import pathlib

import numpy as np
import openmdao.api as om
import pytest

pytest.importorskip("fatpack")

_TOWER_FATIGUE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tower_fatigue_post.py"
_SPEC = importlib.util.spec_from_file_location("tower_fatigue_post", _TOWER_FATIGUE_PATH)
_TOWER_FATIGUE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOWER_FATIGUE_MODULE)
CylinderFatiguePostFrame = _TOWER_FATIGUE_MODULE.CylinderFatiguePostFrame


def _run_problem(
    loads,
    section_D,
    section_t,
    case_probability,
    case_duration,
    case_n_samples=None,
    n_theta=36,
    **component_options,
):
    n_ts, n_sec, n_time = loads["tower_Fz_ts"].shape

    options = {
        "n_ts": n_ts,
        "n_sec": n_sec,
        "n_time": n_time,
        "n_theta": n_theta,
        "check_probability_sum": True,
    }
    options.update(component_options)

    prob = om.Problem()
    prob.model.add_subsystem(
        "fatigue",
        CylinderFatiguePostFrame(**options),
        promotes=["*"],
    )
    prob.setup()

    for name in ["tower_Fx_ts", "tower_Fy_ts", "tower_Fz_ts", "tower_Mx_ts", "tower_My_ts", "tower_Mz_ts"]:
        prob.set_val(name, loads[name])
    prob.set_val("section_D", section_D, units="m")
    prob.set_val("section_t", section_t, units="m")
    prob.set_val("case_probability", case_probability)
    prob.set_val("case_duration", case_duration, units="s")
    if case_n_samples is not None:
        prob.set_val("case_n_samples", case_n_samples)

    prob.run_model()
    return prob


def _zero_loads(n_ts, n_sec, n_time):
    shape = (n_ts, n_sec, n_time)
    return {
        "tower_Fx_ts": np.zeros(shape),
        "tower_Fy_ts": np.zeros(shape),
        "tower_Fz_ts": np.zeros(shape),
        "tower_Mx_ts": np.zeros(shape),
        "tower_My_ts": np.zeros(shape),
        "tower_Mz_ts": np.zeros(shape),
    }


def _sinusoidal_my_loads(n_ts, n_sec, n_time, amplitude=5.0e6):
    loads = _zero_loads(n_ts, n_sec, n_time)
    time = np.linspace(0.0, 2.0 * np.pi, n_time, endpoint=False)
    moment = amplitude * np.sin(time)

    for i_case in range(n_ts):
        for i_sec in range(n_sec):
            loads["tower_My_ts"][i_case, i_sec, :] = moment

    return loads


def _component(**options):
    defaults = {
        "n_ts": 1,
        "n_sec": 1,
        "n_time": 3,
    }
    defaults.update(options)
    return CylinderFatiguePostFrame(**defaults)


def test_single_slope_cycles_to_failure_matches_legacy_formula():
    sn_intercept = 2.5e12
    sn_slope = 3.5
    ranges = np.array([20.0, 50.0, 100.0])
    comp = _component(
        sn_model="single_slope",
        sn_intercept=sn_intercept,
        sn_slope=sn_slope,
        apply_thickness_correction=False,
    )

    n_fail = comp._cycles_to_failure(ranges, thickness=0.05)

    np.testing.assert_allclose(n_fail, sn_intercept / ranges**sn_slope)


def test_dnv_bilinear_cycles_to_failure_is_continuous_at_switch():
    dnv_m1 = 3.0
    dnv_loga1 = 12.164
    dnv_m2 = 5.0
    dnv_n_switch = 1.0e7
    comp = _component(
        sn_model="dnv_bilinear",
        dnv_m1=dnv_m1,
        dnv_loga1=dnv_loga1,
        dnv_m2=dnv_m2,
        dnv_loga2=None,
        dnv_n_switch=dnv_n_switch,
        apply_thickness_correction=False,
    )
    a1 = 10.0**dnv_loga1
    delta_switch = (a1 / dnv_n_switch) ** (1.0 / dnv_m1)

    n_fail = comp._cycles_to_failure(np.array([delta_switch]), thickness=0.05)

    np.testing.assert_allclose(n_fail, np.array([dnv_n_switch]), rtol=1.0e-12)


def test_dnv_bilinear_branch_selection_matches_expected_formulas():
    dnv_m1 = 3.0
    dnv_loga1 = 12.164
    dnv_m2 = 5.0
    dnv_n_switch = 1.0e7
    comp = _component(
        sn_model="dnv_bilinear",
        dnv_m1=dnv_m1,
        dnv_loga1=dnv_loga1,
        dnv_m2=dnv_m2,
        dnv_loga2=None,
        dnv_n_switch=dnv_n_switch,
        apply_thickness_correction=False,
    )
    a1 = 10.0**dnv_loga1
    delta_switch = (a1 / dnv_n_switch) ** (1.0 / dnv_m1)
    a2 = dnv_n_switch * delta_switch**dnv_m2
    ranges = np.array([2.0 * delta_switch, 0.5 * delta_switch])

    n_fail = comp._cycles_to_failure(ranges, thickness=0.05)

    expected = np.array(
        [
            a1 * ranges[0] ** (-dnv_m1),
            a2 * ranges[1] ** (-dnv_m2),
        ]
    )
    np.testing.assert_allclose(n_fail, expected, rtol=1.0e-12)


def test_thickness_correction_never_reduces_effective_stress_range():
    stress_range = np.array([100.0])
    sn_intercept = 1.0e12
    sn_slope = 3.0
    comp = _component(
        sn_model="single_slope",
        sn_intercept=sn_intercept,
        sn_slope=sn_slope,
        apply_thickness_correction=True,
        t_ref=0.025,
        k_thickness=0.20,
    )

    n_fail_thin = comp._cycles_to_failure(stress_range, thickness=0.015)
    n_fail_ref = sn_intercept / stress_range**sn_slope
    n_fail_thick = comp._cycles_to_failure(stress_range, thickness=0.050)

    np.testing.assert_allclose(n_fail_thin, n_fail_ref)
    assert n_fail_thick[0] < n_fail_ref[0]


def test_zero_loads_give_zero_damage():
    prob = _run_problem(
        loads=_zero_loads(n_ts=2, n_sec=3, n_time=16),
        section_D=np.array([4.0, 5.0, 6.0]),
        section_t=np.array([0.03, 0.04, 0.05]),
        case_probability=np.array([0.4, 0.6]),
        case_duration=np.array([600.0, 600.0]),
    )

    np.testing.assert_allclose(prob.get_val("damage_25y"), 0.0)
    np.testing.assert_allclose(prob.get_val("constr_fatigue"), 0.0)


@pytest.mark.parametrize("sn_model", ["single_slope", "dnv_bilinear"])
def test_bending_damage_is_positive_for_sn_models(sn_model):
    fatigue_design_factor = 1.7
    prob = _run_problem(
        loads=_sinusoidal_my_loads(n_ts=1, n_sec=1, n_time=64),
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([64.0]),
        sn_model=sn_model,
        fatigue_design_factor=fatigue_design_factor,
    )

    damage = prob.get_val("damage_25y")
    constr = prob.get_val("constr_fatigue")

    assert damage[0] > 0.0
    np.testing.assert_allclose(constr, fatigue_design_factor * damage)


def test_simple_bending_only_damage_is_positive_and_peaks_at_x_axis():
    n_theta = 36
    prob = _run_problem(
        loads=_sinusoidal_my_loads(n_ts=1, n_sec=1, n_time=64),
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
        n_theta=n_theta,
    )

    damage_theta = prob.get_val("damage_25y_theta")[0, :]
    assert prob.get_val("damage_25y")[0] > 0.0
    assert prob.get_val("constr_fatigue")[0] > 0.0

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    peak_indices = np.flatnonzero(np.isclose(damage_theta, np.max(damage_theta), rtol=1.0e-6))
    peak_angles = theta[peak_indices]

    assert np.any(np.isclose(peak_angles, 0.0, atol=2.0 * np.pi / n_theta))
    assert np.any(np.isclose(peak_angles, np.pi, atol=2.0 * np.pi / n_theta))


def test_probability_scaling_matches_equivalent_single_case():
    one_case = _run_problem(
        loads=_sinusoidal_my_loads(n_ts=1, n_sec=1, n_time=64),
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
    )

    two_case = _run_problem(
        loads=_sinusoidal_my_loads(n_ts=2, n_sec=1, n_time=64),
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([0.25, 0.75]),
        case_duration=np.array([600.0, 600.0]),
    )

    np.testing.assert_allclose(two_case.get_val("damage_25y"), one_case.get_val("damage_25y"), rtol=1.0e-12)
    np.testing.assert_allclose(
        two_case.get_val("constr_fatigue"),
        one_case.get_val("constr_fatigue"),
        rtol=1.0e-12,
    )


def test_case_n_samples_excludes_temporal_zero_padding_from_damage():
    n_valid = 64
    n_time = 96
    loads_padded = _zero_loads(n_ts=1, n_sec=1, n_time=n_time)
    loads_unpadded = _zero_loads(n_ts=1, n_sec=1, n_time=n_valid)
    moment = 5.0e6 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_valid, endpoint=False))
    loads_padded["tower_My_ts"][0, 0, :n_valid] = moment
    loads_unpadded["tower_My_ts"][0, 0, :] = moment

    padded = _run_problem(
        loads=loads_padded,
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
        case_n_samples=np.array([n_valid]),
    )
    unpadded = _run_problem(
        loads=loads_unpadded,
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
    )

    np.testing.assert_allclose(padded.get_val("damage_25y"), unpadded.get_val("damage_25y"), rtol=1.0e-12)
    np.testing.assert_allclose(
        padded.get_val("damage_25y_theta"),
        unpadded.get_val("damage_25y_theta"),
        rtol=1.0e-12,
    )


def test_zero_probability_cases_do_not_contribute_to_damage():
    loads = _zero_loads(n_ts=2, n_sec=1, n_time=64)
    loads["tower_My_ts"][0, 0, :] = 5.0e6 * np.sin(np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False))
    loads["tower_My_ts"][1, 0, :] = 5.0e7 * np.sin(np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False))

    two_case = _run_problem(
        loads=loads,
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0, 0.0]),
        case_duration=np.array([600.0, 600.0]),
    )
    one_case = _run_problem(
        loads={name: values[:1, :, :] for name, values in loads.items()},
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
    )

    np.testing.assert_allclose(two_case.get_val("damage_25y"), one_case.get_val("damage_25y"), rtol=1.0e-12)


@pytest.mark.parametrize(
    "case_n_samples, message",
    [
        (np.array([-1]), "case_n_samples entries must be non-negative."),
        (np.array([65]), "case_n_samples entries cannot exceed n_time."),
    ],
)
def test_invalid_case_n_samples_raise_value_error(case_n_samples, message):
    prob = om.Problem()
    prob.model.add_subsystem(
        "fatigue",
        CylinderFatiguePostFrame(n_ts=1, n_sec=1, n_time=64),
        promotes=["*"],
    )
    prob.setup()
    loads = _sinusoidal_my_loads(n_ts=1, n_sec=1, n_time=64)
    for name in ["tower_Fx_ts", "tower_Fy_ts", "tower_Fz_ts", "tower_Mx_ts", "tower_My_ts", "tower_Mz_ts"]:
        prob.set_val(name, loads[name])
    prob.set_val("section_D", np.array([5.0]), units="m")
    prob.set_val("section_t", np.array([0.04]), units="m")
    prob.set_val("case_probability", np.array([1.0]))
    prob.set_val("case_duration", np.array([600.0]), units="s")
    prob.set_val("case_n_samples", case_n_samples)

    with pytest.raises(ValueError, match=message):
        prob.run_model()


def test_thinner_section_has_larger_damage_than_thicker_section():
    prob = _run_problem(
        loads=_sinusoidal_my_loads(n_ts=1, n_sec=2, n_time=64),
        section_D=np.array([5.0, 5.0]),
        section_t=np.array([0.02, 0.08]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
    )

    damage = prob.get_val("damage_25y")
    assert damage[0] > damage[1]
