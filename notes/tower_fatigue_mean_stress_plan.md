# Tower fatigue — mean-stress correction: theory, decision, and implementation plan

Context: `weis/aeroelasticse/tower_fatigue_post.py` (`TowerFatiguePostFrame`).
Status as of 2026-07-03: the component applies **no mean-stress correction**.
This document records whether that is correct and how to add a correction if/when needed.

---

## 1. Decision / stance (read this first)

**For a welded steel tower analysed with DNV-RP-C203 S-N curves, applying NO
mean-stress correction is the code-correct default, not a loss of accuracy.**

Reason: DNV-RP-C203 design S-N curves are derived from constant-amplitude tests
on *as-welded* specimens that already contain high tensile residual stresses
(near yield). The curves therefore implicitly assume the worst-case mean stress,
and the standard states that for as-welded joints the mean stress has little
effect and no correction should generally be applied. Adding a Goodman
correction on top of an as-welded DNV curve is **not** what the standard intends
and can be either double-conservative or non-conservative.

So: keep `mean_stress_model = "none"` as the default. Add a correction only as an
**optional, physically-justified** path for the cases in §2.

## 2. When mean stress actually matters

- **Parent / base material** (non-welded, machined) — mean stress matters.
- **Stress-relieved (PWHT) welds** — DNV allows a *partial* benefit for the
  compressive portion of the cycle (§3a). This is the DNV-sanctioned mechanism.
- **Non-steel** details (aluminium, composite, cast) — different rules entirely.

For an as-welded steel tower (the usual case here), none of these apply and the
current behaviour is correct.

## 3. Two implementable models

### (a) DNV-RP-C203 compressive / stress-relief reduction (recommended for welded steel)
Only valid for stress-relieved welds or base material. From a rainflow cycle
with range `Δ` and mean `m`: `σmax = m + Δ/2`, `σmin = m − Δ/2`. Effective range:

- fully tensile  (`σmin ≥ 0`):        `Δ_eff = σmax − σmin  (= Δ)`
- fully compressive (`σmax ≤ 0`):     `Δ_eff = 0.6 · (σmax − σmin)`
- partial (`σmin < 0 < σmax`):        `Δ_eff = σmax − 0.6 · σmin`  (σmin < 0 ⇒ adds 0.6·|σmin|)

Example: cycle +100→−100 MPa ⇒ Δ=200 ⇒ Δ_eff = 100 − 0.6·(−100) = 160 MPa.
Factor 0.6 is the DNV default; expose it as a parameter. Gate behind a PWHT flag.

### (b) Goodman / Haigh equivalent fully-reversed stress (for un-welded/wrought material only)
`σ_eq = Δ / (1 − σm / σu)`, i.e. `fatpack.find_goodman_equivalent_stress(S, Sm, Sult)`.
This is what pCrunch offers (`goodman_correction=True`). Requires an ultimate
stress `Sult`. **Not** endorsed by DNV for as-welded joints; clamp/skip for
compressive mean to avoid non-conservatism.

## 4. Implementation steps (default-off, no regression to existing results)

1. **Return means from rainflow.** In `_rainflow_ranges_counts`, call
   `fatpack.find_rainflow_ranges(stress, k=..., return_means=True)` and return
   `(ranges, means, counts)`. (Today it discards means.)
2. **New options / inputs:**
   - option `mean_stress_model` ∈ {`none`, `dnv_stress_relief`, `goodman`}, default `none`.
   - inputs `sn_ult_stress` (Pa; needed for goodman), `mean_stress_factor` (default 0.6),
     wired through `fatigue_ivc` in `glue_code.py` and declared in `modeling_schema.yaml`
     under `TowerFatigue:` alongside the S-N params.
3. **Apply order in `damage_from_stress_timeseries`:**
   `raw ranges+means` → (Pa→MPa) → **mean-stress correction → effective range** →
   **thickness correction** `(t_eff/t_ref)^k` → S-N → Miner.
   (Do mean correction first: it is a property of the cycle; the thickness factor
   then scales the resulting effective range.)
4. **Keep `none` bit-identical** to today so the regression baseline (see the
   test plan) is unchanged when the feature is off.

## 5. Validation / tests

- Unit test on synthetic signals:
  - `none` path reproduces stored golden damage (no change).
  - `goodman`: a tension-tension (positive mean) cycle set gives **higher**
    equivalent range / damage than the same range at zero mean.
  - `dnv_stress_relief`: a fully-compressive cycle set gives **lower** damage
    (factor-0.6 range) than the same tensile cycle.
- Confirm units: ranges in MPa, `Sult` in MPa inside the S-N evaluation.

## 6. Effort / risk

~Half a day. Low risk: the whole feature is gated behind a default-`none`
option, so existing tower-fatigue numbers and the smoke test are unaffected
until a user opts in. The scientifically important point is §1: for as-welded
steel the honest answer is that **no correction is the correct model**, so this
is an optional extension for base-material / PWHT / non-steel details, not a bug
fix.
