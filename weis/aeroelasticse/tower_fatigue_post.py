"""
Tower fatigue post-processing from sectional load time series.

This module intentionally lives in WEIS/QBtoWEIS rather than WISDEM.  The
component receives continuous numeric OpenMDAO inputs from the QBlade load
component and does not read QBlade ``chan_time`` dictionaries or discrete
payloads.

The current fatigue calculation reconstructs normal stress from axial force
``Fz`` and biaxial bending moments ``Mx`` and ``My``. ``Fx``, ``Fy``, and
``Mz`` are not included in the current implementation.

Limitations of the current implementation:
- no shear fatigue;
- no torsional fatigue;
- no mean-stress correction;
- no full multiaxial fatigue model.

The default S-N model, ``single_slope``, is a simple preliminary model using
``N = A / Delta_sigma_eff**m``.  The ``dnv_bilinear`` option is reserved for a
future configurable DNV-RP-C203-style bilinear S-N curve.  DNV coefficients
must be selected consistently with the welded detail class and stress unit.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from openmdao.api import ExplicitComponent
import wisdem.commonse.utilities as util
from wisdem.commonse.cylinder_member import get_nfull
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


class TowerFatiguePostComp(ExplicitComponent):
    """
    OpenMDAO component for tower fatigue post-processing.

    This component does not run QBlade.
    It receives frozen QBlade tower load time series as continuous OpenMDAO inputs
    and recomputes tower fatigue damage from the current tower geometry.
    """

    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        modopt = self.options["modeling_options"]

        n_height_tow = modopt["WISDEM"]["TowerSE"]["n_height"]
        n_full_tow = get_nfull(
            n_height_tow,
            nref=modopt["WISDEM"]["TowerSE"]["n_refine"],
        )

        fatigue_options = modopt["QBlade"].get("tower_fatigue", {})
        self.n_sec_tower_fatigue = n_full_tow - 1
        self.n_theta_tower_fatigue = fatigue_options.get("n_theta", 36)

        self.add_input(
            "z_full",
            val=np.zeros(n_full_tow),
            units="m",
            desc="Full refined tower z-grid from TowerSE.",
        )

        self.add_input(
            "outer_diameter_full",
            val=np.zeros(n_full_tow),
            units="m",
            desc="Full refined tower outer diameter from TowerSE.",
        )

        self.add_input(
            "t_full",
            val=np.zeros(n_full_tow - 1),
            units="m",
            desc="Full refined tower wall thickness by section from TowerSE.",
        )

        n_cases_fat = self.options["modeling_options"]["DLC_driver"]["n_cases"]
        n_stations_fat = 11
        n_time_fat = fatigue_options.get("n_time_max", 92500)

        self.add_input( "tower_fatigue_Fz_ts", val=np.zeros((n_cases_fat, n_stations_fat, n_time_fat)), units="N")
        self.add_input("tower_fatigue_Mx_ts", val=np.zeros((n_cases_fat, n_stations_fat, n_time_fat)), units="N*m")
        self.add_input("tower_fatigue_My_ts", val=np.zeros((n_cases_fat, n_stations_fat, n_time_fat)), units="N*m")
        self.add_input("tower_fatigue_n_time", val=np.zeros(n_cases_fat))
        self.add_input("tower_fatigue_active", val=np.zeros(n_cases_fat))
        self.add_input("tower_fatigue_probability", val=np.zeros(n_cases_fat))
        self.add_input("tower_fatigue_duration", val=np.zeros(n_cases_fat), units="s")
        self.add_input("tower_fatigue_case_id", val=-np.ones(n_cases_fat))
        self.add_output("tower_fatigue_damage_25y", val=np.zeros(self.n_sec_tower_fatigue), desc="Maximum lifetime Miner damage over theta by tower section.")
        self.add_output("tower_fatigue_constr", val=np.zeros(self.n_sec_tower_fatigue), desc="Tower fatigue utilization constraint by section.")
        self.add_output("tower_fatigue_damage_25y_theta", val=np.zeros((self.n_sec_tower_fatigue, self.n_theta_tower_fatigue)), desc="Lifetime Miner damage by tower section and circumferential point.")

        self.declare_partials(
            of=[
                "tower_fatigue_damage_25y",
                "tower_fatigue_constr",
                "tower_fatigue_damage_25y_theta",
            ],
            wrt=[
                "z_full",
                "outer_diameter_full",
                "t_full",
            ],
            method="fd",
        )
        for name in [
            "tower_fatigue_Fz_ts",
            "tower_fatigue_Mx_ts",
            "tower_fatigue_My_ts",
            "tower_fatigue_n_time",
            "tower_fatigue_active",
            "tower_fatigue_probability",
            "tower_fatigue_duration",
            "tower_fatigue_case_id",
        ]:
            try:
                self.declare_partials(
                    of=[
                        "tower_fatigue_damage_25y",
                        "tower_fatigue_constr",
                        "tower_fatigue_damage_25y_theta",
                    ],
                    wrt=name,
                    dependent=False,
                )
            except TypeError:
                pass
        
    def _tower_fatigue_helper_options(self):
        fatigue_options = self.options["modeling_options"]["QBlade"].get(
            "tower_fatigue", {}
        )
        return {
            "n_theta": fatigue_options.get("n_theta", 36),
            "lifetime_years": fatigue_options.get("lifetime_years", 25.0),
            "fatigue_design_factor": fatigue_options.get(
                "fatigue_design_factor", 1.0
            ),
            "sn_model": fatigue_options.get("sn_model", "single_slope"),
            "sn_slope": fatigue_options.get("sn_slope", 3.0),
            "sn_intercept": fatigue_options.get("sn_intercept", 10.0**12.164),
            "dnv_m1": fatigue_options.get("dnv_m1", 3.0),
            "dnv_loga1": fatigue_options.get("dnv_loga1", 12.010),
            "dnv_m2": fatigue_options.get("dnv_m2", 5.0),
            "dnv_loga2": fatigue_options.get("dnv_loga2", 15.350),
            "dnv_n_switch": fatigue_options.get("dnv_n_switch", 1.0e7),
            "stress_unit": fatigue_options.get("stress_unit", "MPa"),
            "apply_thickness_correction": fatigue_options.get(
                "apply_thickness_correction", True
            ),
            "t_ref": fatigue_options.get("t_ref", 0.025),
            "k_thickness": fatigue_options.get("k_thickness", 0.20),
            "rainflow_bins": fatigue_options.get("rainflow_bins", 128),
            "normalize_probabilities": fatigue_options.get(
                "normalize_probabilities", False
            ),
            "check_probability_sum": fatigue_options.get(
                "check_probability_sum", False
            ),
        }

    def compute(self, inputs, outputs):
        outputs["tower_fatigue_damage_25y"] = np.zeros_like(
            outputs["tower_fatigue_damage_25y"]
        )
        outputs["tower_fatigue_constr"] = np.zeros_like(
            outputs["tower_fatigue_constr"]
        )
        outputs["tower_fatigue_damage_25y_theta"] = np.zeros_like(
            outputs["tower_fatigue_damage_25y_theta"]
        )

        active = np.asarray(inputs["tower_fatigue_active"], dtype=float)
        probabilities = np.asarray(inputs["tower_fatigue_probability"], dtype=float)
        durations = np.asarray(inputs["tower_fatigue_duration"], dtype=float)
        n_time_vec = np.asarray(inputs["tower_fatigue_n_time"], dtype=float)
        case_ids = np.asarray(inputs["tower_fatigue_case_id"], dtype=float)

        metadata = {
            "tower_fatigue_active": active,
            "tower_fatigue_probability": probabilities,
            "tower_fatigue_duration": durations,
            "tower_fatigue_n_time": n_time_vec,
            "tower_fatigue_case_id": case_ids,
        }
        for name, values in metadata.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")

        fatigue_options = self._tower_fatigue_helper_options()
        active_mask = active > 0.5
        if not np.any(active_mask):
            raise RuntimeError("Tower fatigue post-processing received no active QBlade fatigue cases.")

        n_time_alloc = inputs["tower_fatigue_Fz_ts"].shape[2]
        active_probabilities = probabilities[active_mask]
        active_durations = durations[active_mask]
        active_n_time = n_time_vec[active_mask]

        if np.any(active_probabilities < 0.0):
            bad_cases = [
                (int(i), int(round(case_ids[i])))
                for i in np.where(active_mask & (probabilities < 0.0))[0]
            ]
            raise ValueError(
                "Active tower_fatigue_probability entries must be non-negative. "
                f"Bad (slot, case_id): {bad_cases}"
            )
        if np.any(active_durations <= 0.0):
            bad_cases = [
                (int(i), int(round(case_ids[i])))
                for i in np.where(active_mask & (durations <= 0.0))[0]
            ]
            raise ValueError(
                "Active tower_fatigue_duration entries must be positive. "
                f"Bad (slot, case_id): {bad_cases}"
            )
        if np.any(active_n_time < 3.0):
            bad_cases = [
                (int(i), int(round(case_ids[i])))
                for i in np.where(active_mask & (n_time_vec < 3.0))[0]
            ]
            raise ValueError(
                "Active tower_fatigue_n_time entries must be at least 3. "
                f"Bad (slot, case_id): {bad_cases}"
            )
        if np.any(active_n_time > n_time_alloc):
            bad_cases = [
                (int(i), int(round(case_ids[i])))
                for i in np.where(active_mask & (n_time_vec > n_time_alloc))[0]
            ]
            raise ValueError(
                f"Active tower_fatigue_n_time entries must not exceed allocated time dimension "
                f"{n_time_alloc}. Bad (slot, case_id): {bad_cases}"
            )

        fatigue_probabilities = prepare_tower_fatigue_probabilities(
            active_probabilities,
            fatigue_options,
        )

        z_full = np.asarray(inputs["z_full"], dtype=float)
        outer_diameter_full = np.asarray(inputs["outer_diameter_full"], dtype=float)
        t_full = np.asarray(inputs["t_full"], dtype=float)

        if not np.all(np.isfinite(z_full)):
            raise ValueError("Tower fatigue geometry input z_full must contain only finite values.")
        if not np.all(np.isfinite(outer_diameter_full)):
            raise ValueError("Tower fatigue geometry input outer_diameter_full must contain only finite values.")
        if not np.all(np.isfinite(t_full)):
            raise ValueError("Tower fatigue geometry input t_full must contain only finite values.")

        z_sec, _ = util.nodal2sectional(z_full)
        section_D, _ = util.nodal2sectional(outer_diameter_full)
        section_t = t_full

        if not np.all(np.isfinite(z_sec)):
            raise ValueError("Tower fatigue sectional z grid must contain only finite values.")
        if not np.all(np.diff(z_sec) > 0.0):
            raise ValueError("Tower fatigue sectional z grid must be strictly increasing.")
        if z_sec[-1] <= z_sec[0]:
            raise ValueError("Tower fatigue sectional z grid must have positive height span.")

        if section_D.shape != section_t.shape:
            raise ValueError(
                "Tower fatigue geometry mismatch: section_D and section_t "
                f"must have the same shape, got {section_D.shape} and {section_t.shape}."
            )
        if not np.all(np.isfinite(section_D)):
            raise ValueError("Tower fatigue section_D must contain only finite values.")
        if not np.all(np.isfinite(section_t)):
            raise ValueError("Tower fatigue section_t must contain only finite values.")
        try:
            tower_tube_section_properties(section_D, section_t)
        except ValueError as exc:
            raise ValueError(f"Invalid tower fatigue tube geometry: {exc}") from exc

        z_target = (z_sec - z_sec[0]) / (z_sec[-1] - z_sec[0])
        tower_grid = np.linspace(0.0, 1.0, 11)
        damage_theta = np.zeros_like(outputs["tower_fatigue_damage_25y_theta"])
        active_indices = np.where(active_mask)[0]
        n_processed = 0

        for i_local, i_case in enumerate(active_indices):
            case_id = int(round(case_ids[i_case]))
            n_time = int(round(n_time_vec[i_case]))

            Fz_station = inputs["tower_fatigue_Fz_ts"][i_case, :, :n_time]
            Mx_station = inputs["tower_fatigue_Mx_ts"][i_case, :, :n_time]
            My_station = inputs["tower_fatigue_My_ts"][i_case, :, :n_time]

            case_section_loads = {}

            for load_key, station_values in [
                ("Fz", Fz_station),
                ("Mx", Mx_station),
                ("My", My_station),
            ]:
                expected_shape = (11, n_time)
                if station_values.shape != expected_shape:
                    raise ValueError(
                        f"Tower fatigue load {load_key} for active case slot {i_case} "
                        f"(case_id={case_id}) must have shape {expected_shape}, "
                        f"got {station_values.shape}."
                    )
                if not np.all(np.isfinite(station_values)):
                    raise ValueError(
                        f"Tower fatigue load {load_key} for active case slot {i_case} "
                        f"(case_id={case_id}) contains non-finite values."
                    )
                interpolator = PchipInterpolator(
                    tower_grid,
                    station_values,
                    axis=0,
                )
                case_section_loads[load_key] = interpolator(z_target)

            add_tower_fatigue_case_damage(
                damage_theta,
                case_section_loads["Fz"],
                case_section_loads["Mx"],
                case_section_loads["My"],
                section_D,
                section_t,
                fatigue_probabilities[i_local],
                durations[i_case],
                fatigue_options,
            )
            n_processed += 1

        if n_processed == 0:
            raise RuntimeError("Tower fatigue post-processing did not process any active QBlade fatigue cases.")

        damage_theta, damage_sec, constr_fatigue = finalize_tower_fatigue_outputs(
            damage_theta,
            fatigue_options,
        )

        outputs["tower_fatigue_damage_25y_theta"] = damage_theta
        outputs["tower_fatigue_damage_25y"] = damage_sec
        outputs["tower_fatigue_constr"] = constr_fatigue
