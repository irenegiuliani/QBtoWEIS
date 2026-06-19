"""
Tower fatigue post-processing from sectional load time series.

This module intentionally lives in WEIS/QBtoWEIS rather than WISDEM.  The
component only uses OpenMDAO inputs passed to it and does not depend on QBlade
or WISDEM internals.

The current fatigue calculation reconstructs normal stress from axial force
``Fz`` and biaxial bending moments ``Mx`` and ``My``.  ``Fx``, ``Fy``, and
``Mz`` are accepted as inputs for interface consistency with sectional QBlade
loads and for future shear/torsion fatigue extensions.

The default S-N model, ``single_slope``, is a simple preliminary model using
``N = A / Delta_sigma_eff**m``.  The ``dnv_bilinear`` option is reserved for a
future configurable DNV-RP-C203-style bilinear S-N curve.  DNV coefficients
must be selected consistently with the welded detail class and stress unit.
"""

import numpy as np

try:
    import fatpack
except ImportError as err:
    raise ImportError(
        "Tower fatigue post-processing requires fatpack for rainflow counting. "
        "pCrunch normally depends on fatpack, so installing pCrunch's "
        "dependencies should provide it."
    ) from err


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0

TOWER_FATIGUE_DEFAULTS = {
    "n_theta": 36,
    "lifetime_years": 25.0,
    "fatigue_design_factor": 1.0,
    "sn_model": "single_slope",
    "sn_slope": 3.0,
    "sn_intercept": 10.0**12.164,
    "dnv_m1": 3.0,
    "dnv_loga1": 12.010,
    "dnv_m2": 5.0,
    "dnv_loga2": 15.350,
    "dnv_n_switch": 1.0e7,
    "stress_unit": "MPa",
    "apply_thickness_correction": True,
    "t_ref": 0.025,
    "k_thickness": 0.20,
    "rainflow_bins": 128,
    "normalize_probabilities": False,
    "check_probability_sum": False,
}


def _fatigue_option(fatigue_options, name):
    if fatigue_options is None:
        return TOWER_FATIGUE_DEFAULTS[name]
    try:
        return fatigue_options[name]
    except KeyError:
        return TOWER_FATIGUE_DEFAULTS[name]


def tower_tube_section_properties(D, t):
    """
    Return circular tube area, bending inertia, torsion constant, and radius.
    """
    D = np.asarray(D, dtype=float)
    t = np.asarray(t, dtype=float)

    if np.any(D <= 0.0):
        raise ValueError("section_D entries must be positive.")
    if np.any(t <= 0.0):
        raise ValueError("section_t entries must be positive.")
    if np.any(2.0 * t >= D):
        raise ValueError("section_t must be less than section_D / 2 for a hollow circular tube.")

    Di = D - 2.0 * t
    R = 0.5 * D
    A = 0.25 * np.pi * (D**2 - Di**2)
    I = np.pi / 64.0 * (D**4 - Di**4)
    J = np.pi / 32.0 * (D**4 - Di**4)

    return A, I, J, R


def stress_to_sn_unit(sigma_pa, fatigue_options=None):
    """
    Convert stress from Pa to the unit expected by the S-N curve.
    """
    if _fatigue_option(fatigue_options, "stress_unit") == "MPa":
        return sigma_pa * 1.0e-6
    return sigma_pa


def cycles_to_failure(stress_ranges, thickness, fatigue_options=None):
    """
    Return cycles to failure for stress ranges according to the selected S-N model.
    """
    stress_ranges = np.asarray(stress_ranges, dtype=float)

    if _fatigue_option(fatigue_options, "apply_thickness_correction"):
        thickness_factor = (
            max(float(thickness) / _fatigue_option(fatigue_options, "t_ref"), 1.0)
            ** _fatigue_option(fatigue_options, "k_thickness")
        )
    else:
        thickness_factor = 1.0

    stress_ranges_eff = stress_ranges * thickness_factor
    stress_ranges_eff = np.maximum(stress_ranges_eff, 1.0e-16)

    sn_model = _fatigue_option(fatigue_options, "sn_model")
    if sn_model == "single_slope":
        A = _fatigue_option(fatigue_options, "sn_intercept")
        m = _fatigue_option(fatigue_options, "sn_slope")
        if A <= 0.0:
            raise ValueError("sn_intercept must be positive for the single_slope S-N model.")
        if m <= 0.0:
            raise ValueError("sn_slope must be positive for the single_slope S-N model.")
        return A / stress_ranges_eff**m

    if sn_model == "dnv_bilinear":
        m1 = _fatigue_option(fatigue_options, "dnv_m1")
        loga1 = _fatigue_option(fatigue_options, "dnv_loga1")
        m2 = _fatigue_option(fatigue_options, "dnv_m2")
        loga2 = _fatigue_option(fatigue_options, "dnv_loga2")
        n_switch = _fatigue_option(fatigue_options, "dnv_n_switch")

        if m1 <= 0.0:
            raise ValueError("dnv_m1 must be positive for the dnv_bilinear S-N model.")
        if m2 <= 0.0:
            raise ValueError("dnv_m2 must be positive for the dnv_bilinear S-N model.")
        if n_switch <= 0.0:
            raise ValueError("dnv_n_switch must be positive for the dnv_bilinear S-N model.")

        a1 = 10.0**loga1
        if a1 <= 0.0:
            raise ValueError("10**dnv_loga1 must be positive for the dnv_bilinear S-N model.")

        stress_switch = (a1 / n_switch) ** (1.0 / m1)
        if loga2 is None:
            a2 = n_switch * stress_switch**m2
        else:
            a2 = 10.0**loga2
        if a2 <= 0.0:
            raise ValueError("The low-stress S-N intercept a2 must be positive for the dnv_bilinear S-N model.")

        n_fail = np.empty_like(stress_ranges_eff)
        high_stress = stress_ranges_eff >= stress_switch
        n_fail[high_stress] = a1 * stress_ranges_eff[high_stress] ** (-m1)
        n_fail[~high_stress] = a2 * stress_ranges_eff[~high_stress] ** (-m2)
        return n_fail

    raise ValueError(f"Unknown sn_model '{sn_model}'. Expected 'single_slope' or 'dnv_bilinear'.")


def damage_from_stress_timeseries(stress, thickness, fatigue_options=None):
    """
    Compute unscaled Miner damage for one stress time series.
    """
    stress = np.asarray(stress, dtype=float)
    stress = stress[np.isfinite(stress)]
    if stress.size < 3 or np.allclose(stress, stress[0]):
        return 0.0

    stress_sn = stress_to_sn_unit(stress, fatigue_options)

    try:
        ranges = fatpack.find_rainflow_ranges(stress_sn)
    except (ValueError, IndexError):
        return 0.0
    ranges = np.asarray(ranges, dtype=float)
    ranges = ranges[np.isfinite(ranges) & (ranges > 0.0)]
    if ranges.size == 0:
        return 0.0

    counts, range_bins = fatpack.find_range_count(ranges, _fatigue_option(fatigue_options, "rainflow_bins"))
    counts = np.asarray(counts, dtype=float)
    range_bins = np.asarray(range_bins, dtype=float)
    valid = (
        np.isfinite(counts)
        & np.isfinite(range_bins)
        & (counts > 0.0)
        & (range_bins > 0.0)
    )
    if not np.any(valid):
        return 0.0

    n_fail = cycles_to_failure(range_bins[valid], thickness, fatigue_options)
    return float(np.sum(counts[valid] / n_fail))


def prepare_tower_fatigue_probabilities(probabilities, fatigue_options=None):
    probabilities = np.asarray(probabilities, dtype=float).copy()
    if np.any(probabilities < 0.0):
        raise ValueError("case_probability entries must be non-negative.")

    prob_sum = float(np.sum(probabilities))
    if prob_sum <= 0.0:
        raise ValueError("case_probability must contain positive probability mass.")
    if _fatigue_option(fatigue_options, "normalize_probabilities"):
        probabilities /= prob_sum
    elif _fatigue_option(fatigue_options, "check_probability_sum") and not np.isclose(
        prob_sum, 1.0, rtol=1.0e-3, atol=1.0e-6
    ):
        raise ValueError(
            "case_probability must sum to 1.0 when check_probability_sum is True. "
            "Set normalize_probabilities=True to normalize internally."
        )
    return probabilities


def add_tower_fatigue_case_damage(
    damage_theta,
    Fz_case,
    Mx_case,
    My_case,
    section_D,
    section_t,
    case_probability,
    case_duration,
    fatigue_options,
):
    """
    Add one tower fatigue load case to damage_theta in place.

    This is a plain NumPy helper, not an OpenMDAO component. It accepts local
    case arrays with shape (n_sec, n_time_case) and deliberately avoids full
    (n_ts, n_sec, n_time) load arrays or (n_sec, n_time, n_theta) stress tensors.
    """
    if case_probability < 0.0:
        raise ValueError("case_probability entries must be non-negative.")
    if case_duration <= 0.0:
        raise ValueError("case_duration entries must be positive.")
    if case_probability <= 0.0:
        return

    damage_theta = np.asarray(damage_theta)
    Fz_case = np.asarray(Fz_case)
    Mx_case = np.asarray(Mx_case)
    My_case = np.asarray(My_case)
    section_D = np.asarray(section_D, dtype=float)
    section_t = np.asarray(section_t, dtype=float)

    n_sec, n_theta = damage_theta.shape
    if Fz_case.ndim != 2:
        raise ValueError("Fz_case, Mx_case, and My_case must have shape (n_sec, n_time_case).")
    expected_load_shape = (n_sec, Fz_case.shape[1])
    if (
        Fz_case.shape != expected_load_shape
        or Mx_case.shape != expected_load_shape
        or My_case.shape != expected_load_shape
    ):
        raise ValueError("Fz_case, Mx_case, and My_case must have shape (n_sec, n_time_case).")
    if section_D.shape != (n_sec,):
        raise ValueError("section_D must have shape (n_sec,).")
    if section_t.shape != (n_sec,):
        raise ValueError("section_t must have shape (n_sec,).")
    if Fz_case.shape[1] < 3:
        return

    A, I, _, R = tower_tube_section_properties(section_D, section_t)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    scale_to_life = (
        case_probability
        * _fatigue_option(fatigue_options, "lifetime_years")
        * SECONDS_PER_YEAR
        / case_duration
    )

    for i_sec in range(n_sec):
        Fz = Fz_case[i_sec, :]
        Mx = Mx_case[i_sec, :]
        My = My_case[i_sec, :]

        axial_stress = Fz / A[i_sec]
        for i_theta in range(n_theta):
            stress = (
                axial_stress
                - Mx * R[i_sec] * sin_theta[i_theta] / I[i_sec]
                + My * R[i_sec] * cos_theta[i_theta] / I[i_sec]
            )
            damage_theta[i_sec, i_theta] += scale_to_life * damage_from_stress_timeseries(
                stress, section_t[i_sec], fatigue_options
            )


def finalize_tower_fatigue_outputs(damage_theta, fatigue_options):
    fatigue_design_factor = _fatigue_option(fatigue_options, "fatigue_design_factor")
    damage_sec = np.max(damage_theta, axis=1)
    constr_fatigue = fatigue_design_factor * damage_sec
    return damage_theta, damage_sec, constr_fatigue
