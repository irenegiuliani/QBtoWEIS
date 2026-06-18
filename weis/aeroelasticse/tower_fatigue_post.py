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
import openmdao.api as om

try:
    import fatpack
except ImportError as err:
    raise ImportError(
        "CylinderFatiguePostFrame requires fatpack for rainflow counting. "
        "pCrunch normally depends on fatpack, so installing pCrunch's "
        "dependencies should provide it."
    ) from err


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


class CylinderFatiguePostFrame(om.ExplicitComponent):
    """
    Compute tower fatigue utilization from sectional load time series.

    For each tower section and angular point, this component reconstructs the
    signed axial-plus-bending normal stress,

        sigma = Fz / A - Mx * R * sin(theta) / I + My * R * cos(theta) / I

    then performs rainflow counting and accumulates Miner damage over the
    supplied load cases.  The output damage is scaled to the configured design
    lifetime and the fatigue constraint is the damage multiplied by the fatigue
    design factor.

    The default ``single_slope`` S-N model is the simple preliminary model
    currently used by this component.  ``sn_intercept`` is the intercept ``A``
    in ``N = A / Delta_sigma_eff**m`` and is not interpreted as ``log10(a)``.
    The dnv_bilinear option implements a configurable DNV-RP-C203-style bilinear S-N curve. 
    The coefficients must be selected consistently with the welded detail class, 
    environmental condition, and stress unit.

    The stress basis is normal stress from ``Fz``, ``Mx``, and ``My``.  ``Fx``,
    ``Fy``, and ``Mz`` are kept in the interface for consistency and future
    shear/torsion extensions.

    Time-history arrays may be padded to fixed OpenMDAO dimensions.  The
    ``case_n_samples`` input identifies the physically valid portion of each
    case for rainflow counting, and defaults to the full time-series length to
    preserve backward compatibility when the input is not connected.
    Padded samples are not part of the stress history used for rainflow.
    ``case_duration`` remains the analyzed physical duration used only for
    lifetime scaling.
    """

    def initialize(self):
        self.options.declare("n_ts", types=int, desc="Number of time-series load cases.")
        self.options.declare("n_sec", types=int, desc="Number of tower sections.")
        self.options.declare("n_time", types=int, desc="Number of samples in each time series.")
        self.options.declare("n_theta", default=36, types=int, desc="Number of circumferential fatigue points.")
        self.options.declare("lifetime_years", default=25.0, types=(int, float), desc="Design lifetime.")
        self.options.declare(
            "fatigue_design_factor",
            default=1.0,
            types=(int, float),
            desc="Multiplier applied to fatigue damage for the constraint.",
        )
        self.options.declare(
            "sn_model",
            default="single_slope",
            values=("single_slope", "dnv_bilinear"),
            desc="S-N curve model used for fatigue damage calculation.",
        )
        self.options.declare("sn_slope", default=3.0, types=(int, float), desc="S-N curve slope m.")
        self.options.declare(
            "sn_intercept",
            default=10.0**12.164,
            types=(int, float),
            desc="S-N intercept A in N = A / S**m form.",
        )
        self.options.declare("dnv_m1", default=3.0, types=(int, float))
        self.options.declare("dnv_loga1", default=12.010, types=(int, float)) #curve E from DNV-RP-C203 standard
        self.options.declare("dnv_m2", default=5.0, types=(int, float))
        self.options.declare("dnv_loga2", default=15.350, allow_none=True)
        self.options.declare("dnv_n_switch", default=1.0e7, types=(int, float))
        self.options.declare(
            "stress_unit",
            default="MPa",
            values=("MPa", "Pa"),
            desc="Stress unit expected by the S-N curve.",
        )
        self.options.declare(
            "apply_thickness_correction",
            default=True,
            types=bool,
            desc="Apply thickness correction to stress ranges.",
        )
        self.options.declare("t_ref", default=0.025, types=(int, float), desc="Reference thickness in meters.")
        self.options.declare("k_thickness", default=0.20, types=(int, float), desc="Thickness correction exponent.")
        self.options.declare(
            "rainflow_bins",
            default=128,
            types=int,
            desc="Number of bins used to discretize rainflow ranges.",
        )
        self.options.declare(
            "normalize_probabilities",
            default=False,
            types=bool,
            desc="Normalize nonzero case probabilities before damage weighting.",
        )
        self.options.declare(
            "check_probability_sum",
            default=False,
            types=bool,
            desc="Raise an error if probabilities do not sum to one unless normalization is enabled.",
        )

    def setup(self):
        n_ts = self.options["n_ts"]
        n_sec = self.options["n_sec"]
        n_time = self.options["n_time"]
        n_theta = self.options["n_theta"]

        ts_shape = (n_ts, n_sec, n_time)

        self.add_input("tower_Fx_ts", val=np.zeros(ts_shape), units="N", desc="Tower sectional x-force time series.")
        self.add_input("tower_Fy_ts", val=np.zeros(ts_shape), units="N", desc="Tower sectional y-force time series.")
        self.add_input("tower_Fz_ts", val=np.zeros(ts_shape), units="N", desc="Tower sectional axial force time series.")
        self.add_input("tower_Mx_ts", val=np.zeros(ts_shape), units="N*m", desc="Tower sectional x-moment time series.")
        self.add_input("tower_My_ts", val=np.zeros(ts_shape), units="N*m", desc="Tower sectional y-moment time series.")
        self.add_input("tower_Mz_ts", val=np.zeros(ts_shape), units="N*m", desc="Tower sectional z-moment time series.")
        self.add_input("section_D", val=np.ones(n_sec), units="m", desc="Tower section outer diameter.")
        self.add_input("section_t", val=0.01 * np.ones(n_sec), units="m", desc="Tower section wall thickness.")
        self.add_input("case_probability", val=np.ones(n_ts) / max(n_ts, 1), desc="Probability weight for each case.")
        self.add_input("case_duration", val=600.0 * np.ones(n_ts), units="s", desc="Analyzed duration for each case.")
        self.add_input(
            "case_n_samples",
            val=n_time * np.ones(n_ts),
            desc="Number of valid samples in each padded time series.",
        )

        self.add_output("constr_fatigue", val=np.zeros(n_sec), desc="Fatigue utilization constraint by section.")
        self.add_output("damage_25y", val=np.zeros(n_sec), desc="Maximum 25-year Miner damage over theta by section.")
        self.add_output(
            "damage_25y_theta",
            val=np.zeros((n_sec, n_theta)),
            desc="25-year Miner damage by section and circumferential point.",
        )

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        n_ts = self.options["n_ts"]
        n_sec = self.options["n_sec"]
        n_theta = self.options["n_theta"]

        probabilities = np.asarray(inputs["case_probability"], dtype=float).copy()
        durations = np.asarray(inputs["case_duration"], dtype=float)
        case_n_samples = np.asarray(inputs["case_n_samples"], dtype=int)

        if np.any(probabilities < 0.0):
            raise ValueError("case_probability entries must be non-negative.")
        if np.any(durations <= 0.0):
            raise ValueError("case_duration entries must be positive.")
        if np.any(case_n_samples < 0):
            raise ValueError("case_n_samples entries must be non-negative.")
        if np.any(case_n_samples > self.options["n_time"]):
            raise ValueError("case_n_samples entries cannot exceed n_time.")

        prob_sum = float(np.sum(probabilities))
        if prob_sum <= 0.0:
            raise ValueError("case_probability must contain positive probability mass.")
        if self.options["normalize_probabilities"]:
            probabilities /= prob_sum
        elif self.options["check_probability_sum"] and not np.isclose(prob_sum, 1.0, rtol=1.0e-3, atol=1.0e-6):
            raise ValueError(
                "case_probability must sum to 1.0 when check_probability_sum is True. "
                "Set normalize_probabilities=True to normalize internally."
            )

        D = np.asarray(inputs["section_D"], dtype=float)
        t = np.asarray(inputs["section_t"], dtype=float)
        A, I, _, R = self._tube_section_properties(D, t)

        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        damage_theta = np.zeros((n_sec, n_theta))

        for i_case in range(n_ts):
            if probabilities[i_case] <= 0.0:
                continue

            n_valid = int(case_n_samples[i_case])
            if n_valid < 3:
                continue

            # Padded zeros are numerical artifacts of fixed OpenMDAO output
            # shapes, not QBlade simulation data. Rainflow must use only the
            # real stress history to avoid artificial cycles from the last
            # physical load sample to the padded tail.
            scale_to_life = probabilities[i_case] * self.options["lifetime_years"] * SECONDS_PER_YEAR / durations[i_case]

            for i_sec in range(n_sec):
                Fz = inputs["tower_Fz_ts"][i_case, i_sec, :n_valid]
                Mx = inputs["tower_Mx_ts"][i_case, i_sec, :n_valid]
                My = inputs["tower_My_ts"][i_case, i_sec, :n_valid]

                axial_stress = Fz / A[i_sec]
                for i_theta in range(n_theta):
                    stress = (
                        axial_stress
                        - Mx * R[i_sec] * sin_theta[i_theta] / I[i_sec]
                        + My * R[i_sec] * cos_theta[i_theta] / I[i_sec]
                    )
                    # Rainflow is applied to the signed local normal stress
                    # history, not to separate DELs of the force/moment channels.
                    damage_theta[i_sec, i_theta] += scale_to_life * self._damage_from_stress_timeseries(
                        stress, t[i_sec]
                    )

        damage_sec = np.max(damage_theta, axis=1)

        outputs["damage_25y_theta"] = damage_theta
        outputs["damage_25y"] = damage_sec
        outputs["constr_fatigue"] = self.options["fatigue_design_factor"] * damage_sec

    def _tube_section_properties(self, D, t):
        """
        Return circular tube area, bending inertia, torsion constant, and radius.

        Parameters
        ----------
        D : array_like
            Outer diameter in meters.
        t : array_like
            Wall thickness in meters.

        Returns
        -------
        A : ndarray
            Cross-sectional area in m**2.
        I : ndarray
            Area moment of inertia about either principal bending axis in m**4.
        J : ndarray
            Polar moment of inertia in m**4.
        R : ndarray
            Outer radius in meters.
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

    def _stress_to_sn_unit(self, sigma_pa):
        """
        Convert stress from Pa to the unit expected by the S-N curve.
        """
        if self.options["stress_unit"] == "MPa":
            return sigma_pa * 1.0e-6
        return sigma_pa

    def _damage_from_stress_timeseries(self, stress, thickness):
        """
        Compute unscaled Miner damage for one stress time series.

        The returned value is damage accumulated over the supplied time-series
        duration.  The caller is responsible for probability and lifetime
        scaling.  The supplied stress history is expected to contain only valid
        physical samples.
        """
        stress = np.asarray(stress, dtype=float)
        stress = stress[np.isfinite(stress)]
        if stress.size < 3 or np.allclose(stress, stress[0]):
            return 0.0

        stress_sn = self._stress_to_sn_unit(stress)

        # Count cycles from the signed local normal stress time series.  The
        # fatigue calculation is not based on precomputed DELs of individual
        # force or moment channels.
        try:
            ranges = fatpack.find_rainflow_ranges(stress_sn)
        except (ValueError, IndexError):
            return 0.0
        ranges = np.asarray(ranges, dtype=float)
        ranges = ranges[np.isfinite(ranges) & (ranges > 0.0)]
        if ranges.size == 0:
            return 0.0

        counts, range_bins = fatpack.find_range_count(ranges, self.options["rainflow_bins"])
        counts = np.asarray(counts, dtype=float)
        range_bins = np.asarray(range_bins, dtype=float)
        valid = np.isfinite(counts) & np.isfinite(range_bins) & (counts > 0.0)
        if not np.any(valid):
            return 0.0

        n_fail = self._cycles_to_failure(range_bins[valid], thickness)
        return float(np.sum(counts[valid] / n_fail))

    def _cycles_to_failure(self, stress_ranges, thickness):
        """
        Return cycles to failure for stress ranges according to the selected S-N model.

        Parameters
        ----------
        stress_ranges : array_like
            Stress ranges already converted to the S-N stress unit, usually MPa.
        thickness : float
            Wall thickness in meters.

        Returns
        -------
        N_fail : ndarray
            Cycles to failure for each stress range.
        """
        stress_ranges = np.asarray(stress_ranges, dtype=float)

        if self.options["apply_thickness_correction"]:
            thickness_factor = max(float(thickness) / self.options["t_ref"], 1.0) ** self.options["k_thickness"]
        else:
            thickness_factor = 1.0

        stress_ranges_eff = stress_ranges * thickness_factor
        stress_ranges_eff = np.maximum(stress_ranges_eff, 1.0e-16)

        sn_model = self.options["sn_model"]
        if sn_model == "single_slope":
            A = self.options["sn_intercept"]
            m = self.options["sn_slope"]
            if A <= 0.0:
                raise ValueError("sn_intercept must be positive for the single_slope S-N model.")
            if m <= 0.0:
                raise ValueError("sn_slope must be positive for the single_slope S-N model.")
            return A / stress_ranges_eff**m

        if sn_model == "dnv_bilinear":
            m1 = self.options["dnv_m1"]
            loga1 = self.options["dnv_loga1"]
            m2 = self.options["dnv_m2"]
            loga2 = self.options["dnv_loga2"]
            n_switch = self.options["dnv_n_switch"]

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
