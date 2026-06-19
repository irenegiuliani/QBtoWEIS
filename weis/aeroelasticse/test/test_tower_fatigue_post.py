import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("fatpack")

_TOWER_FATIGUE_PATH = pathlib.Path(__file__).resolve().parents[1] / "tower_fatigue_post.py"
_SPEC = importlib.util.spec_from_file_location("tower_fatigue_post", _TOWER_FATIGUE_PATH)
_TOWER_FATIGUE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOWER_FATIGUE_MODULE)

add_case_damage = _TOWER_FATIGUE_MODULE.add_tower_fatigue_case_damage
cycles_to_failure = _TOWER_FATIGUE_MODULE.cycles_to_failure
finalize_outputs = _TOWER_FATIGUE_MODULE.finalize_tower_fatigue_outputs
prepare_probabilities = _TOWER_FATIGUE_MODULE.prepare_tower_fatigue_probabilities


def _run_cases(
    Fz_cases,
    Mx_cases,
    My_cases,
    section_D,
    section_t,
    case_probability,
    case_duration,
    n_theta=36,
    valid_samples=None,
    **fatigue_options,
):
    options = {"n_theta": n_theta, "check_probability_sum": True}
    options.update(fatigue_options)
    probabilities = prepare_probabilities(case_probability, options)
    durations = np.asarray(case_duration, dtype=float)

    if np.any(durations <= 0.0):
        raise ValueError("case_duration entries must be positive.")

    damage_theta = np.zeros((len(section_D), n_theta))
    if valid_samples is None:
        valid_samples = [Fz.shape[1] for Fz in Fz_cases]

    for Fz, Mx, My, probability, duration, n_valid in zip(
        Fz_cases, Mx_cases, My_cases, probabilities, durations, valid_samples
    ):
        if n_valid < 0:
            raise ValueError("case_n_samples entries must be non-negative.")
        if n_valid > Fz.shape[1]:
            raise ValueError("case_n_samples entries cannot exceed n_time.")
        add_case_damage(
            damage_theta,
            Fz[:, :n_valid],
            Mx[:, :n_valid],
            My[:, :n_valid],
            section_D,
            section_t,
            probability,
            duration,
            options,
        )

    return finalize_outputs(damage_theta, options)


def _zero_case(n_sec, n_time):
    return (
        np.zeros((n_sec, n_time)),
        np.zeros((n_sec, n_time)),
        np.zeros((n_sec, n_time)),
    )


def _sinusoidal_my_case(n_sec, n_time, amplitude=5.0e6):
    Fz, Mx, My = _zero_case(n_sec, n_time)
    time = np.linspace(0.0, 2.0 * np.pi, n_time, endpoint=False)
    My[:, :] = amplitude * np.sin(time)
    return Fz, Mx, My


def test_single_slope_cycles_to_failure_matches_legacy_formula():
    sn_intercept = 2.5e12
    sn_slope = 3.5
    ranges = np.array([20.0, 50.0, 100.0])
    options = {
        "sn_model": "single_slope",
        "sn_intercept": sn_intercept,
        "sn_slope": sn_slope,
        "apply_thickness_correction": False,
    }

    n_fail = cycles_to_failure(ranges, thickness=0.05, fatigue_options=options)

    np.testing.assert_allclose(n_fail, sn_intercept / ranges**sn_slope)


def test_dnv_bilinear_cycles_to_failure_is_continuous_at_switch():
    dnv_m1 = 3.0
    dnv_loga1 = 12.164
    dnv_m2 = 5.0
    dnv_n_switch = 1.0e7
    options = {
        "sn_model": "dnv_bilinear",
        "dnv_m1": dnv_m1,
        "dnv_loga1": dnv_loga1,
        "dnv_m2": dnv_m2,
        "dnv_loga2": None,
        "dnv_n_switch": dnv_n_switch,
        "apply_thickness_correction": False,
    }
    a1 = 10.0**dnv_loga1
    delta_switch = (a1 / dnv_n_switch) ** (1.0 / dnv_m1)

    n_fail = cycles_to_failure(np.array([delta_switch]), thickness=0.05, fatigue_options=options)

    np.testing.assert_allclose(n_fail, np.array([dnv_n_switch]), rtol=1.0e-12)


def test_dnv_bilinear_branch_selection_matches_expected_formulas():
    dnv_m1 = 3.0
    dnv_loga1 = 12.164
    dnv_m2 = 5.0
    dnv_n_switch = 1.0e7
    options = {
        "sn_model": "dnv_bilinear",
        "dnv_m1": dnv_m1,
        "dnv_loga1": dnv_loga1,
        "dnv_m2": dnv_m2,
        "dnv_loga2": None,
        "dnv_n_switch": dnv_n_switch,
        "apply_thickness_correction": False,
    }
    a1 = 10.0**dnv_loga1
    delta_switch = (a1 / dnv_n_switch) ** (1.0 / dnv_m1)
    a2 = dnv_n_switch * delta_switch**dnv_m2
    ranges = np.array([2.0 * delta_switch, 0.5 * delta_switch])

    n_fail = cycles_to_failure(ranges, thickness=0.05, fatigue_options=options)

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
    options = {
        "sn_model": "single_slope",
        "sn_intercept": sn_intercept,
        "sn_slope": sn_slope,
        "apply_thickness_correction": True,
        "t_ref": 0.025,
        "k_thickness": 0.20,
    }

    n_fail_thin = cycles_to_failure(stress_range, thickness=0.015, fatigue_options=options)
    n_fail_ref = sn_intercept / stress_range**sn_slope
    n_fail_thick = cycles_to_failure(stress_range, thickness=0.050, fatigue_options=options)

    np.testing.assert_allclose(n_fail_thin, n_fail_ref)
    assert n_fail_thick[0] < n_fail_ref[0]


def test_zero_loads_give_zero_damage():
    cases = [_zero_case(3, 16), _zero_case(3, 16)]
    _, damage, constr = _run_cases(
        [c[0] for c in cases],
        [c[1] for c in cases],
        [c[2] for c in cases],
        section_D=np.array([4.0, 5.0, 6.0]),
        section_t=np.array([0.03, 0.04, 0.05]),
        case_probability=np.array([0.4, 0.6]),
        case_duration=np.array([600.0, 600.0]),
    )

    np.testing.assert_allclose(damage, 0.0)
    np.testing.assert_allclose(constr, 0.0)


@pytest.mark.parametrize("sn_model", ["single_slope", "dnv_bilinear"])
def test_bending_damage_is_positive_for_sn_models(sn_model):
    Fz, Mx, My = _sinusoidal_my_case(1, 64)
    fatigue_design_factor = 1.7
    _, damage, constr = _run_cases(
        [Fz],
        [Mx],
        [My],
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([64.0]),
        sn_model=sn_model,
        fatigue_design_factor=fatigue_design_factor,
    )

    assert damage[0] > 0.0
    np.testing.assert_allclose(constr, fatigue_design_factor * damage)


def test_simple_bending_only_damage_is_positive_and_peaks_at_x_axis():
    n_theta = 36
    Fz, Mx, My = _sinusoidal_my_case(1, 64)
    damage_theta, damage, constr = _run_cases(
        [Fz],
        [Mx],
        [My],
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
        n_theta=n_theta,
    )

    assert damage[0] > 0.0
    assert constr[0] > 0.0

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    peak_indices = np.flatnonzero(np.isclose(damage_theta[0, :], np.max(damage_theta[0, :]), rtol=1.0e-6))
    peak_angles = theta[peak_indices]

    assert np.any(np.isclose(peak_angles, 0.0, atol=2.0 * np.pi / n_theta))
    assert np.any(np.isclose(peak_angles, np.pi, atol=2.0 * np.pi / n_theta))


def test_probability_scaling_matches_equivalent_single_case():
    Fz, Mx, My = _sinusoidal_my_case(1, 64)
    _, one_damage, one_constr = _run_cases(
        [Fz],
        [Mx],
        [My],
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
    )

    _, two_damage, two_constr = _run_cases(
        [Fz, Fz],
        [Mx, Mx],
        [My, My],
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([0.25, 0.75]),
        case_duration=np.array([600.0, 600.0]),
    )

    np.testing.assert_allclose(two_damage, one_damage, rtol=1.0e-12)
    np.testing.assert_allclose(two_constr, one_constr, rtol=1.0e-12)


def test_valid_sample_count_excludes_temporal_zero_padding_from_damage():
    n_valid = 64
    n_time = 96
    Fz_padded, Mx_padded, My_padded = _zero_case(1, n_time)
    Fz_unpadded, Mx_unpadded, My_unpadded = _zero_case(1, n_valid)
    moment = 5.0e6 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_valid, endpoint=False))
    My_padded[0, :n_valid] = moment
    My_unpadded[0, :] = moment

    padded = _run_cases(
        [Fz_padded],
        [Mx_padded],
        [My_padded],
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
        valid_samples=[n_valid],
    )
    unpadded = _run_cases(
        [Fz_unpadded],
        [Mx_unpadded],
        [My_unpadded],
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
    )

    np.testing.assert_allclose(padded[1], unpadded[1], rtol=1.0e-12)
    np.testing.assert_allclose(padded[0], unpadded[0], rtol=1.0e-12)


def test_zero_probability_cases_do_not_contribute_to_damage():
    Fz1, Mx1, My1 = _sinusoidal_my_case(1, 64, amplitude=5.0e6)
    Fz2, Mx2, My2 = _sinusoidal_my_case(1, 64, amplitude=5.0e7)

    _, two_damage, _ = _run_cases(
        [Fz1, Fz2],
        [Mx1, Mx2],
        [My1, My2],
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0, 0.0]),
        case_duration=np.array([600.0, 600.0]),
    )
    _, one_damage, _ = _run_cases(
        [Fz1],
        [Mx1],
        [My1],
        section_D=np.array([5.0]),
        section_t=np.array([0.04]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
    )

    np.testing.assert_allclose(two_damage, one_damage, rtol=1.0e-12)


@pytest.mark.parametrize(
    "valid_samples, message",
    [
        ([-1], "case_n_samples entries must be non-negative."),
        ([65], "case_n_samples entries cannot exceed n_time."),
    ],
)
def test_invalid_valid_sample_count_raises_value_error(valid_samples, message):
    Fz, Mx, My = _sinusoidal_my_case(1, 64)

    with pytest.raises(ValueError, match=message):
        _run_cases(
            [Fz],
            [Mx],
            [My],
            section_D=np.array([5.0]),
            section_t=np.array([0.04]),
            case_probability=np.array([1.0]),
            case_duration=np.array([600.0]),
            valid_samples=valid_samples,
        )


def test_thinner_section_has_larger_damage_than_thicker_section():
    Fz, Mx, My = _sinusoidal_my_case(2, 64)
    _, damage, _ = _run_cases(
        [Fz],
        [Mx],
        [My],
        section_D=np.array([5.0, 5.0]),
        section_t=np.array([0.02, 0.08]),
        case_probability=np.array([1.0]),
        case_duration=np.array([600.0]),
    )

    assert damage[0] > damage[1]
