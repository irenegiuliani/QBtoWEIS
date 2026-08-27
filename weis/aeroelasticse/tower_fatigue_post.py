"""Tower fatigue post-processing for QBtoWEIS.

The OpenMDAO component receives TowerSE geometry and lightweight metadata for
saved aeroelastic time-series files. ``tower_fatigue_load_channels`` maps each
solver-grid position to its Fz, Mx, and My keys and must provide explicit
``scale_to_si`` factors. Optional ``units`` metadata is informational only.

Loads and reconstructed stresses use SI units; stress ranges are converted
from Pa to MPa before evaluating the S-N curve. Supplied case probabilities are
used without normalization or redistribution. The calculation intentionally
does not trim transients, apply mean-stress correction, impose an endurance
cutoff, or infer a Weibull distribution.
"""

import concurrent.futures
import os
from pathlib import Path

import fatpack
import numpy as np
import openmdao.api as om
import pandas as pd

import wisdem.commonse.cross_sections as cs
import wisdem.commonse.utilities as util


def _parse_tower_fatigue_load_channels(tower_fatigue_load_channels):
    """Parse and validate tower load-channel metadata."""
    if not isinstance(tower_fatigue_load_channels, (list, tuple)):
        raise ValueError("tower_fatigue_load_channels must be a list or tuple of dictionaries.")
    if len(tower_fatigue_load_channels) < 2:
        raise ValueError(
            "tower_fatigue_load_channels must contain at least two tower grid "
            "points for interpolation."
        )

    tower_grid = []
    load_key_map = []
    load_scale_map = []

    for i_grid, channel in enumerate(tower_fatigue_load_channels):
        if not isinstance(channel, dict):
            raise ValueError(
                "Each item in tower_fatigue_load_channels must be a dictionary. "
                f"Item {i_grid} has type {type(channel)}."
            )
        if "twr_sec_pos" not in channel:
            raise KeyError(
                f"Item {i_grid} in tower_fatigue_load_channels is missing "
                "the 'twr_sec_pos' field."
            )
        if "keys" not in channel:
            raise KeyError(
                f"Item {i_grid} in tower_fatigue_load_channels is missing the 'keys' field."
            )

        twr_sec_pos = float(channel["twr_sec_pos"])
        if not np.isfinite(twr_sec_pos):
            raise ValueError(f"'twr_sec_pos' for tower grid point {i_grid} must be finite.")

        keys = channel["keys"]
        if not isinstance(keys, dict):
            raise ValueError(f"'keys' for tower grid point {i_grid} must be a dictionary.")

        load_keys = {}
        for load_name in ("Fz", "Mx", "My"):
            if load_name not in keys:
                raise KeyError(
                    f"'keys' for tower grid point {i_grid} must contain '{load_name}'."
                )
            key = keys[load_name]
            if key is None or str(key).strip() == "":
                raise ValueError(
                    f"The key for '{load_name}' at tower grid point {i_grid} "
                    "must be a non-empty string."
                )
            load_keys[load_name] = str(key)

        if "scale_to_si" not in channel:
            raise ValueError(
                f"tower_fatigue_load_channels item {i_grid} is missing 'scale_to_si'. "
                "Explicit factors for Fz, Mx, and My are required to convert "
                "the stored loads to N and N*m."
            )

        raw_scales = channel["scale_to_si"]
        if not isinstance(raw_scales, dict):
            raise ValueError(
                f"'scale_to_si' for tower grid point {i_grid} must be a dictionary "
                "with keys 'Fz', 'Mx', 'My'."
            )

        load_scales = {}
        for load_name in ("Fz", "Mx", "My"):
            if load_name not in raw_scales:
                raise KeyError(
                    f"'scale_to_si' for tower grid point {i_grid} must contain '{load_name}'."
                )
            scale = float(raw_scales[load_name])
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"'scale_to_si[{load_name!r}]' for tower grid point {i_grid} "
                    f"must be a finite positive number; got {scale!r}."
                )
            load_scales[load_name] = scale

        tower_grid.append(twr_sec_pos)
        load_key_map.append(load_keys)
        load_scale_map.append(load_scales)

    tower_grid = np.asarray(tower_grid, dtype=float)
    if np.any(~np.isfinite(tower_grid)):
        raise ValueError("tower_fatigue_load_channels contains non-finite positions.")

    return tower_grid, load_key_map, load_scale_map


def _get_tower_fatigue_metadata(discrete_inputs):
    """Validate and collect lightweight time-series metadata."""
    ts_dir = discrete_inputs["tower_fatigue_ts_dir"]
    case_names = list(discrete_inputs["tower_fatigue_case_names"])
    case_probability = np.asarray(
        discrete_inputs["tower_fatigue_case_probability"], dtype=float
    )
    case_files = list(discrete_inputs["tower_fatigue_case_files"])
    load_channels = list(discrete_inputs["tower_fatigue_load_channels"])

    n_cases = len(case_names)
    if len(case_files) != n_cases:
        raise ValueError(
            "tower_fatigue_case_files must have the same length as "
            "tower_fatigue_case_names."
        )
    if case_probability.shape != (n_cases,):
        raise ValueError(
            "tower_fatigue_case_probability must have shape "
            f"{(n_cases,)}, but received {case_probability.shape}."
        )

    tower_grid, load_key_map, load_scale_map = _parse_tower_fatigue_load_channels(
        load_channels
    )
    return {
        "ts_dir": ts_dir,
        "case_names": case_names,
        "case_probability": case_probability,
        "case_files": case_files,
        "tower_grid": tower_grid,
        "load_key_map": load_key_map,
        "load_scale_map": load_scale_map,
    }


def _load_time_series_data(ts_dir, case_file, columns=None):
    """Load one saved time-series file."""
    case_path = Path(case_file)
    if not case_path.is_absolute():
        case_path = Path(ts_dir) / case_path
    if not case_path.exists():
        raise FileNotFoundError(f"Time-series file not found: {case_path}")

    suffix = case_path.suffix.lower()
    if suffix in (".p", ".pkl", ".pickle"):
        data = pd.read_pickle(case_path)
    elif suffix == ".parquet":
        data = pd.read_parquet(case_path, columns=columns)
    elif suffix == ".csv":
        data = pd.read_csv(case_path)
    elif suffix == ".npz":
        with np.load(case_path, allow_pickle=False) as npz_data:
            data = {key: np.asarray(npz_data[key]) for key in npz_data.files}
    else:
        raise ValueError(
            f"Unsupported time-series file format '{suffix}'. "
            "Supported formats are .p, .pkl, .pickle, .parquet, .csv, and .npz."
        )

    return data, case_path


def _extract_time_series_key(data, case_path, key):
    """Extract one key from an already loaded time-series object."""
    if key is None or str(key).strip() == "":
        raise ValueError("key must be a non-empty string.")

    if isinstance(data, pd.DataFrame):
        available_keys = list(data.columns)
        if key not in available_keys:
            raise KeyError(
                f"Key '{key}' not found in {case_path}. "
                f"Available columns are: {available_keys}"
            )
        values = data[key].to_numpy(dtype=float)
    elif isinstance(data, dict):
        available_keys = list(data.keys())
        if key not in available_keys:
            raise KeyError(
                f"Key '{key}' not found in {case_path}. "
                f"Available keys are: {available_keys}"
            )
        values = np.asarray(data[key], dtype=float)
    else:
        raise TypeError(
            f"Unsupported object loaded from {case_path}. "
            "Expected a pandas DataFrame or a dictionary-like object."
        )

    values = np.asarray(values, dtype=float).squeeze()
    if values.ndim != 1:
        raise ValueError(
            f"Key '{key}' in {case_path} must be one-dimensional after "
            f"squeeze, but has shape {values.shape}."
        )
    if np.any(~np.isfinite(values)):
        raise ValueError(f"Key '{key}' in {case_path} contains non-finite values.")

    return values


def _load_time_series_key_from_candidates(data, case_path, keys):
    """Load the first available key from a list of candidate key names."""
    last_error = None
    for key in keys:
        try:
            return _extract_time_series_key(data, case_path, key)
        except KeyError as err:
            last_error = err
    raise KeyError(
        f"None of the candidate keys {keys} was found in {case_path}."
    ) from last_error


def _load_case_tower_loads_on_solver_grid(
    ts_dir, case_file, tower_grid, load_key_map, load_scale_map
):
    """Return time and SI loads shaped ``(n_grid, n_time)`` for one case."""
    case_path_for_suffix = Path(case_file)
    if not case_path_for_suffix.is_absolute():
        case_path_for_suffix = Path(ts_dir) / case_path_for_suffix

    required_load_columns = list(
        dict.fromkeys(
            load_keys[load_name]
            for load_keys in load_key_map
            for load_name in ("Fz", "Mx", "My")
        )
    )

    # Parquet can avoid reading unrelated channels, but the time key is not
    # standardized between existing producers.
    if case_path_for_suffix.suffix.lower() == ".parquet":
        data = None
        case_path = None
        time = None
        last_error = None
        for time_key in ("Time", "time"):
            required_columns = list(
                dict.fromkeys([time_key] + required_load_columns)
            )
            try:
                data, case_path = _load_time_series_data(
                    ts_dir, case_file, columns=required_columns
                )
                time = _extract_time_series_key(data, case_path, time_key)
                break
            except ImportError:
                raise
            except Exception as err:
                last_error = err
        if time is None:
            raise KeyError(
                "Unable to load the required time and tower-load columns "
                f"from {case_path_for_suffix}."
            ) from last_error
    else:
        data, case_path = _load_time_series_data(ts_dir, case_file)
        time = _load_time_series_key_from_candidates(data, case_path, ("Time", "time"))

    time = np.asarray(time, dtype=float).squeeze()
    if time.ndim != 1:
        raise ValueError("Time vector must be one-dimensional.")
    if time.size < 2:
        raise ValueError("Time vector must contain at least two samples.")
    if np.any(~np.isfinite(time)):
        raise ValueError("Time vector contains non-finite values.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Time vector must be strictly increasing.")
    if len(load_key_map) != tower_grid.size:
        raise ValueError("load_key_map must have the same length as tower_grid.")
    if len(load_scale_map) != tower_grid.size:
        raise ValueError("load_scale_map must have the same length as tower_grid.")

    n_grid = tower_grid.size
    n_time = time.size
    Fz_grid = np.empty((n_grid, n_time))
    Mx_grid = np.empty((n_grid, n_time))
    My_grid = np.empty((n_grid, n_time))

    for i_grid, (load_keys, load_scales) in enumerate(zip(load_key_map, load_scale_map)):
        Fz_raw = _extract_time_series_key(data, case_path, load_keys["Fz"])
        Mx_raw = _extract_time_series_key(data, case_path, load_keys["Mx"])
        My_raw = _extract_time_series_key(data, case_path, load_keys["My"])

        for name, values in (("Fz", Fz_raw), ("Mx", Mx_raw), ("My", My_raw)):
            if values.shape != (n_time,):
                raise ValueError(
                    f"{name} time series for tower grid point {i_grid} has shape "
                    f"{values.shape}, but expected {(n_time,)}."
                )

        Fz_grid[i_grid, :] = Fz_raw * load_scales["Fz"]
        Mx_grid[i_grid, :] = Mx_raw * load_scales["Mx"]
        My_grid[i_grid, :] = My_raw * load_scales["My"]

    return time, Fz_grid, Mx_grid, My_grid


def _compute_tower_section_properties(z_full, outer_diameter_full, t_full):
    """Return TowerSE section-center geometry and tube properties."""
    z_full = np.asarray(z_full, dtype=float).copy()
    outer_diameter_full = np.asarray(outer_diameter_full, dtype=float).copy()
    t_full = np.asarray(t_full, dtype=float).copy()

    n_full = z_full.size
    n_sec = n_full - 1

    if outer_diameter_full.size != n_full:
        raise ValueError("outer_diameter_full must have the same length as z_full.")
    if t_full.size != n_sec:
        raise ValueError("t_full must have length n_full - 1.")
    if np.any(~np.isfinite(z_full)):
        raise ValueError("z_full contains non-finite values.")
    if np.any(~np.isfinite(outer_diameter_full)):
        raise ValueError("outer_diameter_full contains non-finite values.")
    if np.any(~np.isfinite(t_full)):
        raise ValueError("t_full contains non-finite values.")

    section_L = np.diff(z_full)
    if np.any(section_L <= 0.0):
        raise ValueError("z_full must be strictly increasing from tower bottom to top.")
    if np.any(outer_diameter_full <= 0.0):
        raise ValueError("outer_diameter_full must be positive at all tower nodes.")
    if np.any(t_full <= 0.0):
        raise ValueError("t_full must be positive in all tower sections.")

    section_z, _ = util.nodal2sectional(z_full)
    section_D, _ = util.nodal2sectional(outer_diameter_full)
    section_r_outer = 0.5 * section_D
    section_r_inner = section_r_outer - t_full

    if np.any(section_r_inner <= 0.0):
        raise ValueError(
            "Invalid tower geometry: each wall thickness must be smaller than "
            "the corresponding sectional outer radius."
        )

    tube = cs.Tube(section_D, t_full, L=section_L)
    return {
        "section_z": section_z,
        "section_L": section_L,
        "section_D": section_D,
        "section_t": t_full,
        "section_r_outer": section_r_outer,
        "section_r_inner": section_r_inner,
        "section_A": tube.Area,
        "section_Asx": tube.Asx,
        "section_Asy": tube.Asy,
        "section_Ixx": tube.Ixx,
        "section_Iyy": tube.Iyy,
        "section_J0": tube.J0,
        "section_Sx": tube.Sx,
        "section_Sy": tube.Sy,
    }


def _get_section_fatigue_data(section_props):
    """Keep only the section arrays needed by fatigue workers."""
    section_Ixx = np.asarray(section_props["section_Ixx"], dtype=float)
    section_Iyy = np.asarray(section_props["section_Iyy"], dtype=float)

    return {
        "A": np.asarray(section_props["section_A"], dtype=float),
        # A circular tube should have equal axes; retain their mean to preserve
        # the existing numerical treatment of any small implementation mismatch.
        "I": 0.5 * (section_Ixx + section_Iyy),
        "R": np.asarray(section_props["section_r_outer"], dtype=float),
        "t": np.asarray(section_props["section_t"], dtype=float),
    }


def _tower_grid_to_z(tower_grid, z_full):
    """Convert normalized or absolute solver-grid positions to tower z."""
    tower_grid = np.asarray(tower_grid, dtype=float).squeeze()
    z_full = np.asarray(z_full, dtype=float).squeeze()

    if tower_grid.ndim != 1:
        raise ValueError("tower_grid must be one-dimensional.")
    if z_full.ndim != 1:
        raise ValueError("z_full must be one-dimensional.")
    if np.any(~np.isfinite(tower_grid)):
        raise ValueError("tower_grid contains non-finite values.")

    # Solver metadata may use either a normalized tower coordinate or absolute z.
    if np.all((tower_grid >= 0.0) & (tower_grid <= 1.0)):
        tower_grid_z = z_full[0] + tower_grid * (z_full[-1] - z_full[0])
    else:
        tower_grid_z = tower_grid

    if np.any(~np.isfinite(tower_grid_z)):
        raise ValueError("tower_grid_z contains non-finite values.")

    return tower_grid_z


def _build_tower_load_interpolation_spec(tower_grid, section_z, z_full):
    """Precompute linear-interpolation indices and weights at section centers."""
    section_z = np.asarray(section_z, dtype=float).squeeze()
    tower_grid_z = _tower_grid_to_z(tower_grid, z_full)

    if section_z.ndim != 1:
        raise ValueError("section_z must be one-dimensional.")

    sort_idx = np.argsort(tower_grid_z)
    tower_grid_z = tower_grid_z[sort_idx]

    if np.any(np.diff(tower_grid_z) <= 0.0):
        raise ValueError(
            "twr_sec_pos entries in tower_fatigue_load_channels must be strictly "
            "increasing after conversion to z-coordinates."
        )
    if section_z[0] < tower_grid_z[0] or section_z[-1] > tower_grid_z[-1]:
        raise ValueError(
            "TowerSE section centers are outside the range covered by "
            "tower_fatigue_load_channels."
        )

    idx = np.searchsorted(tower_grid_z, section_z, side="right") - 1
    idx = np.clip(idx, 0, tower_grid_z.size - 2)
    weight = (section_z - tower_grid_z[idx]) / (
        tower_grid_z[idx + 1] - tower_grid_z[idx]
    )

    return {
        "sort_idx": sort_idx,
        "idx": idx,
        "weight": weight,
        "n_grid": tower_grid_z.size,
    }


def _interpolate_tower_loads_to_sections_from_spec(
    interpolation_spec, Fz_grid, Mx_grid, My_grid
):
    """Return Fz, Mx, and My arrays shaped ``(n_section, n_time)``."""
    interpolated_loads = []
    sort_idx = interpolation_spec["sort_idx"]
    idx = interpolation_spec["idx"]
    weight = interpolation_spec["weight"]
    n_grid = interpolation_spec["n_grid"]

    for load_name, load_grid in (("Fz", Fz_grid), ("Mx", Mx_grid), ("My", My_grid)):
        load_grid = np.asarray(load_grid, dtype=float)
        if load_grid.ndim != 2:
            raise ValueError(f"{load_name}_grid must be two-dimensional.")
        if load_grid.shape[0] != n_grid:
            raise ValueError(
                f"{load_name}_grid first dimension must match the number of tower "
                "load-channel stations."
            )

        load_grid = load_grid[sort_idx, :]
        load_section = (
            (1.0 - weight)[:, None] * load_grid[idx, :]
            + weight[:, None] * load_grid[idx + 1, :]
        )

        if np.any(~np.isfinite(load_section)):
            raise ValueError(f"Interpolated {load_name} contains non-finite values.")

        interpolated_loads.append(load_section)

    return tuple(interpolated_loads)


def _calculate_stress(
    Fz,
    Mx,
    My,
    A,
    I,
    R,
    sin_theta,
    cos_theta,
    section_fatigue_scf=1.0,
):
    """Return one longitudinal normal-stress time series in Pa."""
    Fz = np.asarray(Fz, dtype=float)
    Mx = np.asarray(Mx, dtype=float)
    My = np.asarray(My, dtype=float)

    if Fz.shape != Mx.shape or Fz.shape != My.shape:
        raise ValueError("Fz, Mx, and My must have the same shape.")
    if Fz.ndim != 1:
        raise ValueError("Fz, Mx, and My must be one-dimensional.")
    if A <= 0.0:
        raise ValueError("A must be positive.")
    if I <= 0.0:
        raise ValueError("I must be positive.")
    if R <= 0.0:
        raise ValueError("R must be positive.")
    if section_fatigue_scf <= 0.0:
        raise ValueError("section_fatigue_scf must be positive.")

    sigma = Fz / A - Mx * R * sin_theta / I + My * R * cos_theta / I
    sigma *= section_fatigue_scf

    if np.any(~np.isfinite(sigma)):
        raise ValueError("Calculated stress contains non-finite values.")

    return sigma


def _rainflow_ranges_counts(stress, rainflow_ranges_bins):
    """Return positive binned stress ranges in Pa and their cycle counts."""
    stress = np.asarray(stress, dtype=float).squeeze()

    if stress.ndim != 1:
        raise ValueError("Rainflow input stress must be one-dimensional.")
    if stress.size < 2:
        return np.zeros(0), np.zeros(0)
    if np.any(~np.isfinite(stress)):
        raise ValueError("Rainflow input stress contains non-finite values.")
    if rainflow_ranges_bins <= 0:
        raise ValueError("rainflow_ranges_bins must be positive.")

    try:
        ranges = fatpack.find_rainflow_ranges(stress, k=256)
    except ValueError:
        # fatpack raises for signals with no reversals; they contribute no damage.
        return np.zeros(0), np.zeros(0)

    ranges = np.atleast_1d(np.asarray(ranges, dtype=float).squeeze())
    if ranges.size == 0:
        return np.zeros(0), np.zeros(0)

    ranges = ranges[np.isfinite(ranges) & (ranges > 0.0)]
    if ranges.size == 0:
        return np.zeros(0), np.zeros(0)

    counts, ranges = fatpack.find_range_count(ranges, bins=100)
    counts = np.atleast_1d(np.asarray(counts, dtype=float).squeeze())
    ranges = np.atleast_1d(np.asarray(ranges, dtype=float).squeeze())
    valid = np.isfinite(ranges) & np.isfinite(counts) & (ranges > 0.0) & (counts > 0.0)

    return ranges[valid], counts[valid]


def _damage_from_stress_timeseries(stress, section_t, fatigue_settings):
    """Return simulated-time Palmgren-Miner damage for one stress series."""
    stress = np.asarray(stress, dtype=float).squeeze()

    if stress.ndim != 1:
        raise ValueError("stress must be one-dimensional.")
    if stress.size < 2:
        return 0.0
    if np.any(~np.isfinite(stress)):
        raise ValueError("stress contains non-finite values.")

    stress_ranges_pa, counts = _rainflow_ranges_counts(
        stress, fatigue_settings["rainflow_ranges_bins"]
    )
    if stress_ranges_pa.size == 0:
        return 0.0

    sn_k = fatigue_settings["sn_k_full"]
    sn_tref = fatigue_settings["sn_tref_full"]

    t_eff = max(float(section_t), sn_tref)
    stress_ranges_corr_mpa = stress_ranges_pa * 1.0e-6 * (t_eff / sn_tref) ** sn_k
    valid = stress_ranges_corr_mpa > 0.0
    if not np.any(valid):
        return 0.0

    stress_ranges_corr_mpa = stress_ranges_corr_mpa[valid]
    counts = counts[valid]

    if fatigue_settings["sn_model"] == "linear":
        log_a = fatigue_settings["sn_log_a_full"]
        m = fatigue_settings["sn_m_full"]
        cycles_to_failure = np.power(10.0, log_a) / np.power(stress_ranges_corr_mpa, m)
    elif fatigue_settings["sn_model"] == "bilinear":
        log_a1 = fatigue_settings["sn_log_a1_full"]
        m1 = fatigue_settings["sn_m1_full"]
        log_a2 = fatigue_settings["sn_log_a2_full"]
        m2 = fatigue_settings["sn_m2_full"]
        transition_cycles = fatigue_settings["sn_transition_cycles_full"]

        cycles_branch_1 = np.power(10.0, log_a1) / np.power(
            stress_ranges_corr_mpa, m1
        )
        cycles_branch_2 = np.power(10.0, log_a2) / np.power(
            stress_ranges_corr_mpa, m2
        )
        cycles_to_failure = np.where(
            cycles_branch_1 <= transition_cycles,
            cycles_branch_1,
            cycles_branch_2,
        )
    else:
        raise ValueError(
            f"Unsupported sn_model '{fatigue_settings['sn_model']}'. "
            "Expected 'linear' or 'bilinear'."
        )

    if np.any(cycles_to_failure <= 0.0):
        raise ValueError("Cycles to failure must be positive.")

    return float(np.sum(counts / cycles_to_failure))


def _calculate_damage_for_case(
    Fz_case,
    Mx_case,
    My_case,
    case_probability,
    case_duration,
    section_fatigue_data,
    theta_stress_points,
    fatigue_settings,
):
    """Return lifetime-scaled damage shaped ``(n_section, n_theta)``."""
    Fz_case = np.asarray(Fz_case, dtype=float)
    Mx_case = np.asarray(Mx_case, dtype=float)
    My_case = np.asarray(My_case, dtype=float)

    if Fz_case.shape != Mx_case.shape or Fz_case.shape != My_case.shape:
        raise ValueError("Fz_case, Mx_case, and My_case must have the same shape.")
    if Fz_case.ndim != 2:
        raise ValueError("Fz_case, Mx_case, and My_case must be two-dimensional.")
    if case_duration <= 0.0:
        raise ValueError("case_duration must be positive.")
    if not np.isfinite(case_probability):
        raise ValueError("case_probability must be finite.")
    if case_probability < 0.0:
        raise ValueError("case_probability must be non-negative.")

    n_sec, _ = Fz_case.shape
    A = np.asarray(section_fatigue_data["A"], dtype=float)
    I = np.asarray(section_fatigue_data["I"], dtype=float)
    R = np.asarray(section_fatigue_data["R"], dtype=float)
    section_t = np.asarray(section_fatigue_data["t"], dtype=float)

    for name, values in (("A", A), ("I", I), ("R", R), ("t", section_t)):
        if values.shape != (n_sec,):
            raise ValueError(f"section_fatigue_data['{name}'] must have shape {(n_sec,)}.")
    if np.any(A <= 0.0):
        raise ValueError("section_A must be positive.")
    if np.any(I <= 0.0):
        raise ValueError("section_Ixx and section_Iyy must be positive.")
    if np.any(R <= 0.0):
        raise ValueError("section_r_outer must be positive.")
    if np.any(section_t <= 0.0):
        raise ValueError("section_t must be positive.")

    section_fatigue_scf = fatigue_settings["section_fatigue_scf"]
    if section_fatigue_scf <= 0.0:
        raise ValueError("section_fatigue_scf must be positive.")

    theta = np.asarray(theta_stress_points, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta_stress_points must be one-dimensional.")
    if theta.size < 4:
        raise ValueError("At least four theta stress points are required.")

    if fatigue_settings["design_life"] <= 0.0:
        raise ValueError("design_life must be positive.")

    scale_to_life = case_probability * fatigue_settings["design_life"] / case_duration
    damage_theta = np.zeros((n_sec, theta.size))
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    for i_sec in range(n_sec):
        Fz = Fz_case[i_sec, :]
        Mx = Mx_case[i_sec, :]
        My = My_case[i_sec, :]

        for i_theta in range(theta.size):
            stress = _calculate_stress(
                Fz=Fz,
                Mx=Mx,
                My=My,
                A=A[i_sec],
                I=I[i_sec],
                R=R[i_sec],
                sin_theta=sin_theta[i_theta],
                cos_theta=cos_theta[i_theta],
                section_fatigue_scf=section_fatigue_scf,
            )
            damage_theta[i_sec, i_theta] = (
                scale_to_life
                * _damage_from_stress_timeseries(
                    stress=stress,
                    section_t=section_t[i_sec],
                    fatigue_settings=fatigue_settings,
                )
            )

    return damage_theta


def _get_validated_fatigue_inputs(inputs, sn_model, rainflow_ranges_bins):
    """Return the picklable worker settings and local fatigue design factor."""
    fatigue_settings = {
        "sn_model": sn_model,
        "rainflow_ranges_bins": int(rainflow_ranges_bins),
        "section_fatigue_scf": float(np.atleast_1d(inputs["section_fatigue_scf"])[0]),
        "design_life": float(np.atleast_1d(inputs["design_life"])[0]),
        "sn_k_full": float(np.atleast_1d(inputs["sn_k_full"])[0]),
        "sn_tref_full": float(np.atleast_1d(inputs["sn_tref_full"])[0]),
        "sn_log_a_full": float(np.atleast_1d(inputs["sn_log_a_full"])[0]),
        "sn_m_full": float(np.atleast_1d(inputs["sn_m_full"])[0]),
        "sn_log_a1_full": float(np.atleast_1d(inputs["sn_log_a1_full"])[0]),
        "sn_m1_full": float(np.atleast_1d(inputs["sn_m1_full"])[0]),
        "sn_log_a2_full": float(np.atleast_1d(inputs["sn_log_a2_full"])[0]),
        "sn_m2_full": float(np.atleast_1d(inputs["sn_m2_full"])[0]),
        "sn_transition_cycles_full": float(np.atleast_1d(inputs["sn_transition_cycles_full"])[0]),
    }
    fatigue_design_factor = float(
        np.atleast_1d(inputs["fatigue_design_factor"])[0]
    )

    if fatigue_settings["section_fatigue_scf"] <= 0.0:
        raise ValueError("section_fatigue_scf must be positive.")
    if fatigue_design_factor <= 0.0:
        raise ValueError("fatigue_design_factor must be positive.")
    if fatigue_settings["sn_tref_full"] <= 0.0:
        raise ValueError("sn_tref_full must be positive.")
    if fatigue_settings["design_life"] <= 0.0:
        raise ValueError("design_life must be positive.")
    if fatigue_settings["rainflow_ranges_bins"] <= 0:
        raise ValueError("rainflow_ranges_bins must be positive.")

    if sn_model == "linear":
        if fatigue_settings["sn_m_full"] <= 0.0:
            raise ValueError("sn_m_full must be positive.")
    elif sn_model == "bilinear":
        if fatigue_settings["sn_m1_full"] <= 0.0:
            raise ValueError("sn_m1_full must be positive.")
        if fatigue_settings["sn_m2_full"] <= 0.0:
            raise ValueError("sn_m2_full must be positive.")
        if fatigue_settings["sn_transition_cycles_full"] <= 0.0:
            raise ValueError("sn_transition_cycles_full must be positive.")

    return fatigue_settings, fatigue_design_factor


def _get_requested_n_workers(modeling_options, n_workers, number_of_workers=None):
    """Resolve worker count using modeling options, legacy alias, then default."""
    tower_fatigue_options = modeling_options.get("TowerFatigue", {})
    requested_n_workers = int(
        tower_fatigue_options.get(
            "n_workers",
            tower_fatigue_options.get(
                "number_of_workers",
                n_workers if number_of_workers is None else number_of_workers,
            ),
        )
    )
    if requested_n_workers < 1:
        raise ValueError("TowerFatigue n_workers must be at least 1.")
    return requested_n_workers


def _process_tower_fatigue_case(case_request, shared_payload):
    """Load, interpolate, and fatigue-process one case in one worker process."""
    case_index = case_request.get("case_index")
    case_name = case_request.get("case_name")
    case_file = case_request.get("case_file")

    try:
        case_probability = float(case_request["case_probability"])

        time, Fz_grid, Mx_grid, My_grid = _load_case_tower_loads_on_solver_grid(
            ts_dir=shared_payload["ts_dir"],
            case_file=case_file,
            tower_grid=shared_payload["tower_grid"],
            load_key_map=shared_payload["load_key_map"],
            load_scale_map=shared_payload["load_scale_map"],
        )

        case_duration = float(time[-1] - time[0])
        if case_duration <= 0.0:
            raise ValueError(f"Case duration must be positive for {case_file}.")

        Fz_case, Mx_case, My_case = _interpolate_tower_loads_to_sections_from_spec(
            interpolation_spec=shared_payload["interpolation_spec"],
            Fz_grid=Fz_grid,
            Mx_grid=Mx_grid,
            My_grid=My_grid,
        )

        damage_theta_case = _calculate_damage_for_case(
            Fz_case=Fz_case,
            Mx_case=Mx_case,
            My_case=My_case,
            case_probability=case_probability,
            case_duration=case_duration,
            section_fatigue_data=shared_payload["section_fatigue_data"],
            theta_stress_points=shared_payload["theta_stress_points"],
            fatigue_settings=shared_payload["fatigue_settings"],
        )

        return case_index, damage_theta_case
    except Exception as err:
        raise RuntimeError(
            "Failed to process tower fatigue case "
            f"case_index={case_index}, case_name={case_name!r}, case_file={case_file!r}."
        ) from err


def _run_tower_fatigue_case_workers(
    active_case_requests, shared_payload, n_workers
):
    """Run fatigue cases serially or in a ProcessPoolExecutor."""
    if n_workers < 1:
        raise ValueError("TowerFatigue n_workers must be at least 1.")

    effective_workers = min(
        int(n_workers), len(active_case_requests), os.cpu_count() or 1
    )
    result_by_case = {}

    if effective_workers <= 1:
        for case_request in active_case_requests:
            case_index, damage_theta_case = _process_tower_fatigue_case(
                case_request, shared_payload
            )
            result_by_case[case_index] = damage_theta_case
        return result_by_case, effective_workers

    with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = [
            executor.submit(_process_tower_fatigue_case, case_request, shared_payload)
            for case_request in active_case_requests
        ]
        for future in concurrent.futures.as_completed(futures):
            case_index, damage_theta_case = future.result()
            result_by_case[case_index] = damage_theta_case

    # Results are keyed here and accumulated later in original case order.
    return result_by_case, effective_workers


class TowerFatiguePostFrame(om.ExplicitComponent):
    """Compute lifetime tower fatigue damage from saved load time series.

    Stress is evaluated at ``n_theta`` circumferential points as

    ``Fz / A - Mx * R * sin(theta) / I + My * R * cos(theta) / I``.

    Rainflow ranges are corrected as
    ``Delta_sigma * (max(t, t_ref) / t_ref)**k`` and evaluated with the
    selected linear or bilinear S-N curve. Each case is scaled by
    ``case_probability * design_life / case_duration``.
    """

    def initialize(self):
        self.options.declare("modeling_options", default={}, types=dict)
        self.options.declare("n_full", types=int)
        self.options.declare(
            "n_workers",
            default=1,
            types=int,
            desc="Number of case-level worker processes used for tower fatigue.",
        )
        self.options.declare(
            "number_of_workers",
            default=None,
            allow_none=True,
            types=(int, type(None)),
            desc="Deprecated alias for n_workers.",
        )
        self.options.declare(
            "n_theta",
            default=36,
            types=int,
            desc=(
                "Number of circumferential stress-evaluation points used on "
                "each tower section for fatigue stress reconstruction."
            ),
        )
        self.options.declare(
            "sn_model",
            default="bilinear",
            values=("linear", "bilinear"),
            desc="S-N curve model to be used for fatigue damage evaluation.",
        )
        self.options.declare(
            "rainflow_ranges_bins",
            default=256,
            types=int,
            desc=(
                "Number of load-range bins used by fatpack.find_range_count. "
                "Default 256 to match the pCrunch/NREL rainflow-counting interface."
            ),
        )

    def setup(self):
        n_full = self.options["n_full"]
        n_theta = self.options["n_theta"]
        n_sec = n_full - 1

        if n_theta < 4:
            raise ValueError("n_theta must be at least 4.")

        seconds_per_year = 365.25 * 24.0 * 3600.0

        # Geometry from TowerSE/WISDEM.
        self.add_input("z_full", val=np.zeros(n_full), units="m")
        self.add_input("outer_diameter_full", val=np.zeros(n_full), units="m")
        self.add_input("t_full", val=np.zeros(n_sec), units="m")

        # Lightweight time-series metadata from the aeroelastic workflow.
        self.add_discrete_input("tower_fatigue_ts_dir", val="")
        self.add_discrete_input("tower_fatigue_case_names", val=())
        self.add_discrete_input("tower_fatigue_case_probability", val=())
        self.add_discrete_input("tower_fatigue_case_files", val=())
        self.add_discrete_input("tower_fatigue_load_channels", val=())

        # General fatigue inputs.
        self.add_input("section_fatigue_scf", val=1.0)
        self.add_input("fatigue_design_factor", val=1.0)

        # Thickness correction parameters for the S-N curve.
        # Defaults correspond to DNVGL-RP-C203 curve E in air.
        self.add_input("sn_k_full", val=0.20)
        self.add_input("sn_tref_full", val=0.025, units="m")

        # Linear S-N curve inputs.
        self.add_input("sn_log_a_full", val=12.010)
        self.add_input("sn_m_full", val=3.0)

        # Bilinear S-N curve inputs.
        self.add_input("sn_log_a1_full", val=12.010)
        self.add_input("sn_m1_full", val=3.0)
        self.add_input("sn_log_a2_full", val=15.350)
        self.add_input("sn_m2_full", val=5.0)
        self.add_input("sn_transition_cycles_full", val=1.0e7)

        # Lifetime aggregation input.
        self.add_input("design_life", val=25.0 * seconds_per_year, units="s")

        # Sectional geometry reconstructed internally from TowerSE inputs.
        self.add_output("section_z", val=np.zeros(n_sec), units="m")
        self.add_output("section_L", val=np.zeros(n_sec), units="m")
        self.add_output("section_D", val=np.zeros(n_sec), units="m")
        self.add_output("section_t", val=np.zeros(n_sec), units="m")

        self.add_output("section_r_outer", val=np.zeros(n_sec), units="m")
        self.add_output("section_r_inner", val=np.zeros(n_sec), units="m")

        self.add_output("section_A", val=np.zeros(n_sec), units="m**2")
        self.add_output("section_Asx", val=np.zeros(n_sec), units="m**2")
        self.add_output("section_Asy", val=np.zeros(n_sec), units="m**2")

        self.add_output("section_Ixx", val=np.zeros(n_sec), units="m**4")
        self.add_output("section_Iyy", val=np.zeros(n_sec), units="m**4")
        self.add_output("section_J0", val=np.zeros(n_sec), units="m**4")

        self.add_output("section_Sx", val=np.zeros(n_sec), units="m**3")
        self.add_output("section_Sy", val=np.zeros(n_sec), units="m**3")

        # Circumferential stress-evaluation points.
        self.add_output("theta_stress_points", val=np.zeros(n_theta), units="rad")

        # Fatigue outputs.
        self.add_output("fatigue_damage", val=np.zeros(n_sec))
        self.add_output("constr_fatigue", val=np.zeros(n_sec))

        # Fatigue post-processing includes file reading, rainflow counting, and
        # discontinuous cycle counting. Finite differences are the safest
        # initial choice.
        self.declare_partials("*", "*", method="fd")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute tower section properties, lifetime fatigue damage, and fatigue
        constraints.

        Active time-series cases are independent. They are processed in parallel
        when ``n_workers`` is greater than one, then accumulated in the
        original case order to keep deterministic results.
        """
        n_theta = self.options["n_theta"]
        n_sec = self.options["n_full"] - 1

        if discrete_inputs is None:
            raise ValueError("TowerFatiguePostFrame requires discrete time-series metadata.")

        sn_model = self.options["sn_model"]
        n_workers = _get_requested_n_workers(
            self.options["modeling_options"],
            self.options["n_workers"],
            self.options["number_of_workers"],
        )
        metadata = _get_tower_fatigue_metadata(discrete_inputs)
        fatigue_settings, fatigue_design_factor = _get_validated_fatigue_inputs(
            inputs=inputs,
            sn_model=sn_model,
            rainflow_ranges_bins=self.options["rainflow_ranges_bins"],
        )

        section_props = _compute_tower_section_properties(
            z_full=inputs["z_full"],
            outer_diameter_full=inputs["outer_diameter_full"],
            t_full=inputs["t_full"],
        )

        theta_stress_points = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

        for name, value in section_props.items():
            outputs[name] = value

        outputs["theta_stress_points"] = theta_stress_points

        active_case_requests = []
        for i_case, case_file in enumerate(metadata["case_files"]):
            case_probability = float(metadata["case_probability"][i_case])

            if not np.isfinite(case_probability):
                raise ValueError(f"case_probability for case {i_case} must be finite.")
            if case_probability < 0.0:
                raise ValueError(f"case_probability for case {i_case} must be non-negative.")
            if case_probability == 0.0:
                continue

            active_case_requests.append(
                {
                    "case_index": i_case,
                    "case_name": metadata["case_names"][i_case],
                    "case_file": case_file,
                    "case_probability": case_probability,
                }
            )

        damage_theta = np.zeros((n_sec, n_theta))

        if active_case_requests:
            interpolation_spec = _build_tower_load_interpolation_spec(
                tower_grid=metadata["tower_grid"],
                section_z=section_props["section_z"],
                z_full=inputs["z_full"],
            )
            shared_payload = {
                "ts_dir": metadata["ts_dir"],
                "tower_grid": metadata["tower_grid"],
                "load_key_map": metadata["load_key_map"],
                "load_scale_map": metadata["load_scale_map"],
                "interpolation_spec": interpolation_spec,
                "section_fatigue_data": _get_section_fatigue_data(section_props),
                "theta_stress_points": theta_stress_points,
                "fatigue_settings": fatigue_settings,
            }
            result_by_case, _ = _run_tower_fatigue_case_workers(
                active_case_requests=active_case_requests,
                shared_payload=shared_payload,
                n_workers=n_workers,
            )

            # Futures may finish out of order; sum in the original case order.
            for case_request in active_case_requests:
                damage_theta += result_by_case[case_request["case_index"]]

        fatigue_damage = np.max(damage_theta, axis=1)

        outputs["fatigue_damage"] = fatigue_damage
        outputs["constr_fatigue"] = fatigue_damage * fatigue_design_factor
