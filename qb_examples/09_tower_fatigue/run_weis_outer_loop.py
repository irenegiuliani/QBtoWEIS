"""
Fixed-load iterative tower optimization - multi-iteration outer loop script.

For a single optimization run with frozen loads (no outer iteration), set
  freeze_loads: True
in your modeling options YAML and run run_weis.py as normal. run_weis() will
execute QBlade once, freeze the loads, and then hand control to the optimizer.

Use this script for an Ingersoll/Ning-style outer loop:
  For each outer iteration k:
    1. Run QBlade once with the current tower geometry to compute loads.
    2. Freeze the distributed tower loads.
    3. Run the inner optimizer with loads held constant.
    4. Run QBlade again with the optimized geometry and updated loads.
    5. Check convergence on tower mass; repeat if needed.

Key design decisions:
  - modeling_options["QBlade"]["freeze_loads"] must be True.
  - The base SQL recorder from analysis_options stores the full outer-loop run.
  - Each inner optimization also writes its own SQL recorder:
      <base>_outer_0.sql, <base>_outer_1.sql, ...
  - The inner optimizer for outer_<k> uses loads from outer_<k>/iteration_0/.
  - The fresh QBlade run after an inner optimizer is saved as either
    outer_<k>/iteration_1/ for the final run or outer_<k+1>/iteration_0/ for the
    next frozen-load inner optimization.
"""

import os
import sys
import time

import numpy as np
import openmdao.api as om

from weis.glue_code.runWEIS import run_weis


# ---------------------------------------------------------------------------
# Configuration - edit these to match your run
# ---------------------------------------------------------------------------
run_dir = os.path.dirname(os.path.realpath(__file__))
fname_wt_input = os.path.join(run_dir, "MED-15-300-RWT.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options_MED.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options_MED.yaml")

N_OUTER = 5
OUTER_TOL = 5e-3


def _apply_driver_recording_options(wt_opt, opt_options):
    """Apply the same driver recording options used by the analysis options."""
    wt_opt.driver.recording_options["excludes"] = ["*_df"]
    wt_opt.driver.recording_options["record_constraints"] = True
    wt_opt.driver.recording_options["record_desvars"] = True
    wt_opt.driver.recording_options["record_objectives"] = True

    includes = opt_options["recorder"].get("includes", [])
    if includes:
        wt_opt.driver.recording_options["includes"] = includes


def _add_outer_recorder(wt_opt, opt_options, folder_output, sql_name):
    """Attach a temporary recorder for one frozen-load inner optimization."""
    if not opt_options.get("recorder", {}).get("flag", False):
        return None

    recorder_path = os.path.abspath(os.path.join(folder_output, sql_name))
    recorder = om.SqliteRecorder(recorder_path)
    wt_opt.driver.add_recorder(recorder)
    wt_opt.add_recorder(recorder)
    _apply_driver_recording_options(wt_opt, opt_options)
    recorder.startup(wt_opt.driver, wt_opt.comm)
    recorder.startup(wt_opt, wt_opt.comm)

    print(f"[outer loop] Inner-loop recorder set to: {recorder_path}")
    return recorder


def _remove_recorder(wt_opt, recorder):
    """Remove and shut down one temporary recorder, leaving global recorders alive."""
    if recorder is None:
        return

    for owner in (wt_opt.driver, wt_opt):
        rec_mgr = getattr(owner, "_rec_mgr", None)
        recorders = getattr(rec_mgr, "_recorders", None)
        if recorders is not None and recorder in recorders:
            recorders.remove(recorder)

    try:
        recorder.shutdown()
    except Exception:
        pass


def _prepare_qblade_run(qb_comp, modeling_options, base_qb_dir, outer_k, qb_inumber):
    """Point QBlade at one outer-loop iteration directory."""
    outer_qb_dir = os.path.join(base_qb_dir, f"outer_{outer_k}")
    os.makedirs(outer_qb_dir, exist_ok=True)

    qb_comp.QBLADE_runDirectory = outer_qb_dir
    qb_comp.qb_inumber = qb_inumber
    qb_comp.freeze_mode = False

    qbsim = modeling_options["QBlade"]["simulation"]
    if qbsim.get("DLCGenerator", False) or qbsim.get("WNDTYPE", 0) == 1:
        qb_comp.wind_directory = os.path.join(outer_qb_dir, "wind")
        os.makedirs(qb_comp.wind_directory, exist_ok=True)

    return outer_qb_dir


def _run_fresh_qblade(
    wt_opt,
    qb_comp,
    modeling_options,
    base_qb_dir,
    outer_k,
    qb_inumber,
    label,
):
    """Run QBlade once and leave the resulting loads available for freezing."""
    outer_qb_dir = _prepare_qblade_run(
        qb_comp=qb_comp,
        modeling_options=modeling_options,
        base_qb_dir=base_qb_dir,
        outer_k=outer_k,
        qb_inumber=qb_inumber,
    )

    print(f"[outer {outer_k}] {label} - running QBlade simulation...")
    sys.stdout.flush()
    wt_opt.run_model()

    qb_comp.freeze_mode = True
    print(
        f"[outer {outer_k}] QBlade done. freeze_mode = True. "
        f"Frozen loads saved to {outer_qb_dir}/iteration_{qb_inumber}/frozen_loads.p"
    )
    return outer_qb_dir


def run_outer_loop():
    tt_total = time.time()

    print("\n" + "=" * 70)
    print("OUTER LOOP: setting up OpenMDAO problem (setup_only=True)")
    print("=" * 70 + "\n")

    result = run_weis(
        fname_wt_input,
        fname_modeling_options,
        fname_analysis_options,
        setup_only=True,
    )
    wt_opt, wt_initial, modeling_options, opt_options, myopt, folder_output, rank = result

    if not modeling_options["QBlade"].get("freeze_loads", False):
        raise ValueError(
            "modeling_options['QBlade']['freeze_loads'] must be True to use the "
            "fixed-load outer loop. Set 'freeze_loads: True' in your modeling options YAML."
        )

    qb_comp = wt_opt.model.aeroelastic_qblade
    base_qb_dir = qb_comp.QBLADE_runDirectory

    rec_cfg = opt_options.get("recorder", {})
    rec_fname = rec_cfg.get("file_name", "recorder.sql")
    rec_base, _ = os.path.splitext(rec_fname)

    prev_tower_mass = np.inf
    converged = False
    have_frozen_loads_for_current_outer = False

    for outer_k in range(N_OUTER):
        print("\n" + "=" * 70)
        print(f"OUTER ITERATION {outer_k}")
        print("=" * 70 + "\n")

        tt_outer = time.time()

        # Phase A: make sure outer_k/iteration_0 contains the load snapshot
        # used by the frozen-load inner optimizer.
        if not have_frozen_loads_for_current_outer:
            _run_fresh_qblade(
                wt_opt=wt_opt,
                qb_comp=qb_comp,
                modeling_options=modeling_options,
                base_qb_dir=base_qb_dir,
                outer_k=outer_k,
                qb_inumber=0,
                label="Phase A",
            )
        else:
            qb_comp.freeze_mode = True
            print(
                f"[outer {outer_k}] Phase A - reusing fresh QBlade loads "
                f"already saved in outer_{outer_k}/iteration_0"
            )

        # Phase B: inner optimizer with frozen loads.
        outer_sql = f"{rec_base}outer{outer_k}.sql"
        outer_recorder = _add_outer_recorder(wt_opt, opt_options, folder_output, outer_sql)

        print(f"[outer {outer_k}] Phase B - running inner optimizer (loads frozen)...")
        sys.stdout.flush()
        try:
            wt_opt.run_driver()
            wt_opt.record(f"outer_{outer_k}_inner_complete")
        finally:
            _remove_recorder(wt_opt, outer_recorder)

        curr_tower_mass = float(wt_opt.get_val("towerse.tower_mass")[0])
        if np.isfinite(prev_tower_mass):
            rel_delta = abs(curr_tower_mass - prev_tower_mass) / max(abs(prev_tower_mass), 1.0)
        else:
            rel_delta = np.inf

        will_converge = rel_delta < OUTER_TOL and outer_k > 0
        is_last_outer = outer_k == N_OUTER - 1

        # Phase C: run QBlade with the optimized tower geometry. If another
        # outer iteration is needed, store the result as the next iteration's
        # frozen-load snapshot to avoid an immediate duplicate QBlade run.
        if will_converge or is_last_outer:
            fresh_outer_k = outer_k
            fresh_qb_inumber = 1
            have_frozen_loads_for_current_outer = False
        else:
            fresh_outer_k = outer_k + 1
            fresh_qb_inumber = 0
            have_frozen_loads_for_current_outer = True

        _run_fresh_qblade(
            wt_opt=wt_opt,
            qb_comp=qb_comp,
            modeling_options=modeling_options,
            base_qb_dir=base_qb_dir,
            outer_k=fresh_outer_k,
            qb_inumber=fresh_qb_inumber,
            label="Phase C",
        )
        wt_opt.record(f"outer_{outer_k}_fresh_qblade")

        print(
            f"\n[outer {outer_k}] tower_mass = {curr_tower_mass:.1f} kg "
            f"| delta = {rel_delta:.4f} (tol = {OUTER_TOL})"
        )
        print(f"[outer {outer_k}] elapsed: {time.time() - tt_outer:.1f} s")

        if will_converge:
            print(
                f"\n[outer loop] CONVERGED at outer iteration {outer_k} "
                f"(delta = {rel_delta:.4f} < {OUTER_TOL})"
            )
            converged = True
            break

        prev_tower_mass = curr_tower_mass

    if not converged:
        print(
            f"\n[outer loop] Reached maximum outer iterations ({N_OUTER}). "
            f"Final tower_mass = {prev_tower_mass:.1f} kg"
        )

    if rank == 0:
        from openfast_io.FileTools import save_yaml
        from wisdem.commonse import fileIO
        from wisdem.inputs.validation import simple_types

        froot_out = os.path.join(folder_output, opt_options["general"]["fname_output"])
        modeling_options["General"]["openfast_configuration"]["fst_vt"] = {}
        if not modeling_options["OpenFAST"]["from_openfast"]:
            wt_initial.write_ontology(wt_opt, froot_out)
        wt_initial.write_options(froot_out)

        problem_var_dict = wt_opt.list_driver_vars(
            desvar_opts=["lower", "upper"],
            cons_opts=["lower", "upper", "equals"],
        )
        save_yaml(folder_output, "problem_vars.yaml", simple_types(problem_var_dict))
        fileIO.save_data(froot_out, wt_opt)

    print(f"\n[outer loop] Total run time: {time.time() - tt_total:.1f} s")
    return wt_opt, modeling_options, opt_options


if __name__ == "__main__":
    wt_opt, modeling_options, opt_options = run_outer_loop()