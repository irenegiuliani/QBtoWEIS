"""
Fixed-Load Iterative Tower Optimization — Multi-Iteration Outer Loop Script

For a single optimization run with frozen loads (no outer iteration), set
  freeze_loads: True
in your modeling options YAML and run run_weis.py as normal.  run_weis() will
automatically execute QBlade once, freeze the loads, and then hand control to
the optimizer — no outer loop script required.

Use THIS script when you want the full Ingersoll & Ning outer loop:
  For each outer iteration k:
    1. Run QBlade ONCE with the current tower geometry  →  compute loads
    2. Freeze the distributed tower loads
    3. Run the inner optimizer (COBYLA / SLSQP) with loads held constant
    4. Check convergence on tower mass; repeat if not converged

Key design decisions:
  - modeling_options['QBlade']['freeze_loads'] MUST be True in the modeling YAML.
  - Each outer iteration writes its own SQL recorder file:
      <base>_outer_0.sql, <base>_outer_1.sql, ...
  - QBlade outputs are saved under outer_<k>/iteration_0/ (existing save_iterations
    directory structure is preserved within each outer iteration).
  - Frozen loads are written to outer_<k>/iteration_0/frozen_loads.p as a pickle.
"""

import os
import sys
import time
import numpy as np
import pickle
import openmdao.api as om

from weis.glue_code.runWEIS import run_weis

# ---------------------------------------------------------------------------
# Configuration — edit these to match your run
# ---------------------------------------------------------------------------
run_dir               = os.path.dirname(os.path.realpath(__file__))
fname_wt_input        = os.path.join(run_dir, "IEA-15-240-RWT_VolturnUS-S_rectangular.yaml")
fname_modeling_options = os.path.join(run_dir, "modeling_options_custom_dlc.yaml")
fname_analysis_options = os.path.join(run_dir, "analysis_options_opt.yaml")

N_OUTER      = 5        # maximum number of outer (fixed-load) iterations
OUTER_TOL    = 5e-3     # convergence tolerance on relative tower-mass change

# ---------------------------------------------------------------------------


def _swap_recorder(wt_opt, opt_options, folder_output, new_sql_name):
    """Replace the OpenMDAO driver recorder with a new file for this outer iteration.

    Shuts down all existing recorders on wt_opt.driver and wt_opt, then attaches
    a new SqliteRecorder pointing at new_sql_name inside folder_output.
    Recording options (includes list, record_constraints, etc.) are re-applied from
    opt_options so the new file mirrors what the original analysis_options specified.
    """
    # Shut down and remove existing recorders from driver
    for rec in list(getattr(wt_opt.driver, '_recorders', [])):
        try:
            rec.shutdown()
        except Exception:
            pass
    wt_opt.driver._recorders = []

    # Shut down and remove existing recorders from problem-level
    for rec in list(getattr(wt_opt, '_recorders', [])):
        try:
            rec.shutdown()
        except Exception:
            pass
    wt_opt._recorders = []

    # Attach new recorder if recording is enabled in analysis options
    if opt_options.get("recorder", {}).get("flag", False):
        new_rec = om.SqliteRecorder(os.path.join(folder_output, new_sql_name))
        wt_opt.driver.add_recorder(new_rec)
        wt_opt.add_recorder(new_rec)

        wt_opt.driver.recording_options["excludes"] = ["*_df"]
        wt_opt.driver.recording_options["record_constraints"] = True
        wt_opt.driver.recording_options["record_desvars"] = True
        wt_opt.driver.recording_options["record_objectives"] = True
        includes = opt_options["recorder"].get("includes", [])
        if includes:
            wt_opt.driver.recording_options["includes"] = includes

        print(f"[outer loop] Recorder set to: {os.path.join(folder_output, new_sql_name)}")


def run_outer_loop():

    tt_total = time.time()

    # ------------------------------------------------------------------
    # One-time setup: build and fully initialise the OpenMDAO problem
    # without running model or driver.
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("OUTER LOOP: setting up OpenMDAO problem (setup_only=True)")
    print("="*70 + "\n")

    result = run_weis(
        fname_wt_input,
        fname_modeling_options,
        fname_analysis_options,
        setup_only=True,
    )
    wt_opt, wt_initial, modeling_options, opt_options, myopt, folder_output, rank = result

    if not modeling_options['QBlade'].get('freeze_loads', False):
        raise ValueError(
            "modeling_options['QBlade']['freeze_loads'] must be True to use the "
            "fixed-load outer loop.  Set 'freeze_loads: True' in your modeling options YAML."
        )

    # Get a direct reference to the QBlade component
    qb_comp = wt_opt.model.aeroelastic_qblade

    # Store the base QBlade run directory so we can create per-outer-iteration subdirs
    base_qb_dir = qb_comp.QBLADE_runDirectory

    # Derive recorder base name from analysis options (strip .sql extension)
    rec_cfg = opt_options.get("recorder", {})
    rec_fname = rec_cfg.get("file_name", "recorder.sql")
    rec_base, _ = os.path.splitext(rec_fname)

    prev_tower_mass = np.inf
    converged = False

    # ------------------------------------------------------------------
    # Outer iteration loop
    # ------------------------------------------------------------------
    for outer_k in range(N_OUTER):

        print("\n" + "="*70)
        print(f"OUTER ITERATION {outer_k}")
        print("="*70 + "\n")

        tt_outer = time.time()

        # ---------------------------------------------------------------
        # Redirect QBlade output to outer_k/ subdirectory and reset counter
        # ---------------------------------------------------------------
        outer_qb_dir = os.path.join(base_qb_dir, f"outer_{outer_k}")
        os.makedirs(outer_qb_dir, exist_ok=True)

        qb_comp.QBLADE_runDirectory = outer_qb_dir
        qb_comp.qb_inumber = 0
        qb_comp.freeze_mode = False          # explicitly enable QBlade execution

        # Update wind directory if it is used (DLCGenerator or synthetic wind)
        qbsim = modeling_options['QBlade']['simulation']
        if qbsim.get('DLCGenerator', False) or qbsim.get('WNDTYPE', 0) == 1:
            qb_comp.wind_directory = os.path.join(outer_qb_dir, 'wind')
            os.makedirs(qb_comp.wind_directory, exist_ok=True)

        # ---------------------------------------------------------------
        # Phase A: single QBlade model evaluation to compute loads
        # ---------------------------------------------------------------
        print(f"[outer {outer_k}] Phase A — running QBlade simulation...")
        sys.stdout.flush()
        wt_opt.run_model()

        # Loads are now frozen inside qb_comp._frozen_outputs and saved to disk.
        # Enable freeze mode so all subsequent model evaluations return the snapshot.
        qb_comp.freeze_mode = True
        print(f"[outer {outer_k}] QBlade done. freeze_mode = True. "
              f"Frozen loads saved to {outer_qb_dir}/iteration_0/frozen_loads.p")

        # ---------------------------------------------------------------
        # Phase B: inner optimizer with frozen loads
        # ---------------------------------------------------------------
        # Swap the SQL recorder to a file named for this outer iteration
        outer_sql = f"{rec_base}_outer_{outer_k}.sql"
        _swap_recorder(wt_opt, opt_options, folder_output, outer_sql)

        print(f"[outer {outer_k}] Phase B — running inner optimizer (loads frozen)...")
        sys.stdout.flush()
        wt_opt.run_driver()

        # Flush recorder buffers so data is visible on disk before convergence check
        # wt_opt.record_iteration()
        wt_opt.record(f"outer_{outer_k}_complete")
        for rec in getattr(wt_opt.driver, '_recorders', []):
            try:
                rec._server_queue.put('stop')
            except Exception:
                pass

        # ---------------------------------------------------------------
        # Convergence check on tower mass
        # ---------------------------------------------------------------
        curr_tower_mass = float(wt_opt.get_val('towerse.tower_mass')[0])
        rel_delta = abs(curr_tower_mass - prev_tower_mass) / max(abs(prev_tower_mass), 1.0)

        print(f"\n[outer {outer_k}] tower_mass = {curr_tower_mass:.1f} kg  "
              f"| Δ = {rel_delta:.4f}  (tol = {OUTER_TOL})")
        print(f"[outer {outer_k}] elapsed: {time.time() - tt_outer:.1f} s")

        if rel_delta < OUTER_TOL and outer_k > 0:
            print(f"\n[outer loop] CONVERGED at outer iteration {outer_k} "
                  f"(Δ = {rel_delta:.4f} < {OUTER_TOL})")
            converged = True
            break

        prev_tower_mass = curr_tower_mass

    if not converged:
        print(f"\n[outer loop] Reached maximum outer iterations ({N_OUTER}). "
              f"Final tower_mass = {prev_tower_mass:.1f} kg")

    # ------------------------------------------------------------------
    # Post-processing — write ontology, options, problem_vars, numpy data
    # ------------------------------------------------------------------
    if rank == 0:
        from wisdem.commonse import fileIO
        from openfast_io.FileTools import save_yaml
        from wisdem.inputs.validation import simple_types

        froot_out = os.path.join(folder_output, opt_options['general']['fname_output'])
        modeling_options['General']['openfast_configuration']['fst_vt'] = {}
        if not modeling_options['OpenFAST']['from_openfast']:
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
