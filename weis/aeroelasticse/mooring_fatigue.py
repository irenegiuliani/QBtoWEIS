import logging

import fatpack
import numpy as np


logger = logging.getLogger("wisdem/weis")
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def validate_stations(stations, active=True):
    stations = np.asarray(stations, dtype=float)
    if active and stations.size == 0:
        raise ValueError("MooringsFatigue.stations must be non-empty when fatigue is active.")
    if stations.ndim != 1 or np.any(~np.isfinite(stations)) or np.any((stations < 0.0) | (stations > 1.0)):
        raise ValueError("MooringsFatigue.stations must be a finite one-dimensional list in [0, 1].")
    return stations


def line_local_stations(node1_types, node2_types, stations):
    """Map global anchor-to-fairlead stations to each line's node1-to-node2 coordinate."""
    stations = validate_stations(stations)
    if len(node1_types) != len(node2_types):
        raise ValueError("Mooring endpoint type lists must have the same length.")

    mapped = np.empty((len(node1_types), stations.size))
    for i, (type1, type2) in enumerate(zip(node1_types, node2_types)):
        type1 = str(type1).lower()
        type2 = str(type2).lower()
        fixed1, fixed2 = type1.startswith("fix"), type2.startswith("fix")
        vessel1, vessel2 = type1 == "vessel", type2 == "vessel"
        if not ((fixed1 and vessel2) or (vessel1 and fixed2)):
            raise ValueError(
                f"Mooring line {i + 1} must connect exactly one fixed node and one vessel node; "
                f"got {type1!r} and {type2!r}."
            )
        mapped[i] = stations if fixed1 else 1.0 - stations
    return np.round(mapped, 3)


def moordyn_fatigue_channel(line_number, station, num_segs, tolerance=1.0e-8):
    station = float(validate_stations([station])[0])
    num_segs_float = float(num_segs)
    if not np.isfinite(num_segs_float) or num_segs_float <= 0.0 or not np.isclose(num_segs_float, round(num_segs_float)):
        raise ValueError(f"MoorDyn NumSegs for line {line_number} must be a positive integer; got {num_segs!r}.")
    num_segs = int(round(num_segs_float))
    node_float = station * num_segs
    node = int(round(node_float))
    if not np.isclose(node_float, node, atol=tolerance, rtol=0.0):
        raise ValueError(
            f"Mooring fatigue station {station:g} does not coincide with a MoorDyn node for "
            f"line {line_number} with NumSegs={num_segs}."
        )
    if node == 0:
        return f"ANCHTEN{line_number}"
    if node == num_segs:
        return f"FAIRTEN{line_number}"
    return f"L{line_number}N{node}T"


def select_uls_case_ids(labels, requested_labels=None):
    labels = tuple(str(label) for label in labels)
    requested = tuple(str(label) for label in (requested_labels or []))
    if requested:
        missing = sorted(set(requested) - set(labels))
        if missing:
            raise ValueError(f"Requested mooring ULS DLC labels are not present: {missing}.")
        return tuple(i for i, label in enumerate(labels) if label in requested)
    non_custom = tuple(i for i, label in enumerate(labels) if label != "Custom")
    return non_custom or tuple(i for i, label in enumerate(labels) if label == "Custom")


def fatigue_load_sum(case_histories_kn, probabilities, times, tn_m):
    """Return annualized rainflow range sums for histories shaped [case, line, station, time]."""
    tn_m = float(tn_m)
    if not np.isfinite(tn_m) or tn_m <= 0.0:
        raise ValueError(f"Mooring fatigue tn_m must be finite and positive; got {tn_m!r}.")
    if not (len(case_histories_kn) == len(probabilities) == len(times)):
        raise ValueError("Mooring fatigue histories, probabilities, and time vectors must have equal length.")
    if not case_histories_kn:
        raise ValueError("Mooring fatigue requires at least one positive-probability Custom case.")

    probabilities = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("Mooring fatigue probabilities must be finite and non-negative.")
    if abs(float(np.sum(probabilities)) - 1.0) > 1.0e-3:
        logger.warning(
            "Mooring fatigue Custom DLC probabilities sum to %.6f, not 1.0. "
            "The provided probability mass will be used without normalization.",
            float(np.sum(probabilities)),
        )

    shape = np.asarray(case_histories_kn[0]).shape[:2]
    load_sum = np.zeros(shape)
    for i_case, (histories, probability, time) in enumerate(zip(case_histories_kn, probabilities, times)):
        histories = np.asarray(histories, dtype=float)
        time = np.asarray(time, dtype=float)
        if histories.ndim != 3 or histories.shape[:2] != shape:
            raise ValueError(f"Mooring fatigue case {i_case} has inconsistent history shape {histories.shape}.")
        if time.ndim != 1 or time.size < 2 or np.any(~np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
            raise ValueError(f"Invalid time vector for mooring fatigue case {i_case}.")
        if histories.shape[2] != time.size:
            raise ValueError(f"Mooring fatigue case {i_case} tension and time lengths differ.")
        if np.any(~np.isfinite(histories)):
            raise ValueError(f"Non-finite tension history in mooring fatigue case {i_case}.")
        duration = float(time[-1] - time[0])
        if duration <= 0.0:
            raise ValueError(f"Mooring fatigue simulation duration must be positive for case {i_case}.")
        annual_scale = float(probability) * SECONDS_PER_YEAR / duration
        for index in np.ndindex(shape):
            tension = histories[index]
            if tension.size < 2 or np.ptp(tension) == 0.0:
                continue
            try:
                ranges = np.asarray(fatpack.find_rainflow_ranges(tension, k=256), dtype=float)
            except ValueError:
                continue
            ranges = ranges[np.isfinite(ranges) & (ranges > 0.0)]
            load_sum[index] += annual_scale * np.sum(ranges ** tn_m)
    return load_sum


def fatigue_damage(load_sum, mbl_n, tn_k, tn_m, lifetime_years, fatigue_design_factor):
    load_sum = np.asarray(load_sum, dtype=float)
    mbl_n = np.asarray(mbl_n, dtype=float)
    tn_k = float(tn_k)
    tn_m = float(tn_m)
    lifetime_years = float(lifetime_years)
    fatigue_design_factor = float(fatigue_design_factor)
    if load_sum.ndim != 2 or mbl_n.size != load_sum.shape[0]:
        raise ValueError("Mooring fatigue load sums and MBL line dimensions do not match.")
    if np.any(~np.isfinite(load_sum)):
        raise ValueError("Mooring fatigue load sums must be finite.")
    if np.any(~np.isfinite(mbl_n)) or np.any(mbl_n <= 0.0):
        raise ValueError("All mooring MBL values must be finite and positive.")
    if not np.isfinite(tn_k) or tn_k <= 0.0 or not np.isfinite(tn_m) or tn_m <= 0.0:
        raise ValueError("Mooring fatigue tn_k and tn_m must be finite and positive.")
    if not np.isfinite(lifetime_years) or lifetime_years <= 0.0:
        raise ValueError("Mooring fatigue lifetime must be finite and positive.")
    if not np.isfinite(fatigue_design_factor) or fatigue_design_factor <= 0.0:
        raise ValueError("Mooring fatigue design factor must be finite and positive.")
    mbl_kn = mbl_n * 1.0e-3
    damage = lifetime_years * load_sum / (tn_k * mbl_kn[:, None] ** tn_m)
    return damage, fatigue_design_factor * damage
