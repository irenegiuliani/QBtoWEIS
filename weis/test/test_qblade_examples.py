"""
Local-only regression harness for qb_examples/.

Unlike weis/test/test_examples.py (which drives the OpenFAST-based examples/
directory and is exercised by GitHub Actions), this test is never run in
hosted CI: QBlade is a proprietary, separately licensed shared library
(.dll / .so) that cannot be installed on a GitHub-hosted runner. It can only
be run by a developer, locally, on a machine with a licensed QBlade install.

Setup (one-time, per machine):
    Set the QBLADE_DLL_PATH environment variable to your local QBlade shared
    library path, e.g. (PowerShell)
        $env:QBLADE_DLL_PATH = "C:\\Users\\you\\QBladeCE_2.0.9.5\\QBladeCE_2.0.9.5.dll"
    This overrides modeling_options['General']['qblade_configuration']['path2qb_dll']
    at runtime (see weis/aeroelasticse/openmdao_qblade.py:run_QBLADE), so none
    of the qb_examples modeling-options yaml files need to be edited.

Usage as a before/after regression check:
    - qblade_light_scripts: crash-only checks (does the script complete
      without raising). Results print to stdout / whatever each example
      script itself writes to disk under its own output folder -- eyeball
      before/after if you want to compare these by hand.
    - qblade_regression_scripts: scripts with a reviewed golden-value
      baseline, checked automatically via compare_regression_values() (the
      same mechanism used by test_tower_fatigue_post.py). Currently just
      09_tower_fatigue/run_weis, since that's the example this PR's
      tower-fatigue feature is built around and the one whose baseline has
      actually been run and reviewed. To add another script here, or to
      re-baseline after an intentional change, run:
          python weis/test/test_qblade_examples.py --train
      and review the diff to the resulting .pkl file before committing it.

This test is intentionally skipped whenever QBLADE_DLL_PATH is not set, so
adding it does not affect hosted CI or any machine without QBlade installed.
"""

import os
import sys
import unittest

from weis.test.utils import execute_script, compare_regression_values

this_dir = os.path.dirname(os.path.realpath(__file__))

# Fast, standalone scripts: each runs to completion from scratch with no
# dependency on artifacts from a previous run. These are crash-only checks
# (no golden-value comparison) until a baseline has been captured and
# reviewed for each one -- see qblade_regression_scripts below.
qblade_light_scripts = [
    "00_run_test/weis_driver_oc3",
    "01_iea15mw_pyHams/weis_driver",
    "02_iea15mw/weis_driver",
    "03_turbsim/weis_driver",
    "04_dlc_gen/weis_driver_oc3",
    "05_iea15mw_monopile/weis_driver",
    # "06_bladeOpt/weis_driver",
    "07_iea15mw_rect/weis_driver",
    "08_tower_opt/run_weis",
    # 09_tower_fatigue/run_weis is checked separately below, against a
    # reviewed golden-value baseline, since it's the example this PR's
    # tower-fatigue feature is built around.
]

# Scripts with a reviewed golden-value baseline: {script: (output names to
# check,)}. Add an entry here once you've run a script, looked at its
# outputs, and are confident they're a sane baseline -- then regenerate the
# pickle with `python weis/test/test_qblade_examples.py --train`.
qblade_regression_scripts = {
    "09_tower_fatigue/run_weis": (
        "towerse.tower_mass",
        "tower_fatigue_post.constr_fatigue",
    ),
}


def _extract_regression_values(mod, output_names):
    return {name: mod.wt_opt.get_val(name) for name in output_names}

# Multi-iteration outer-loop scripts (Ingersoll & Ning-style fixed-load outer
# loop, see weis/aeroelasticse/tower_fatigue_post.py): each one runs QBlade
# several times end-to-end and is much slower than qblade_light_scripts.
qblade_outer_loop_scripts = [
    "08_tower_opt/run_weis_outer_loop",
    "09_tower_fatigue/run_weis_outer_loop",
]

# Deliberately excluded: these two scripts require an existing .sql recorder
# file from a *previous* run in the same folder as a precondition, so they
# cannot be run standalone/from-scratch like the scripts above.
#   restart_optimization/weis_driver_oc3
#   start_from_sql/weis_driver


def _qblade_dll_available():
    dll_path = os.environ.get("QBLADE_DLL_PATH")
    return bool(dll_path) and os.path.isfile(dll_path)


_SKIP_REASON = (
    "QBlade examples are local-only: they require a licensed QBlade CE/EE "
    "install and cannot run on hosted CI. Set the QBLADE_DLL_PATH environment "
    "variable to your local QBlade .dll/.so path to enable this test."
)


@unittest.skipUnless(_qblade_dll_available(), _SKIP_REASON)
class TestQBladeExamples(unittest.TestCase):

    def test_light_scripts(self):
        for ks, s in enumerate(qblade_light_scripts):
            with self.subTest(f"Running: {s}", i=ks):
                try:
                    execute_script(s, examples_root="qb_examples")
                    self.assertTrue(True)
                except Exception:
                    self.assertEqual(s, "Success")

    @unittest.skipUnless(
        "RUN_QBLADE_OUTER_LOOP" in os.environ,
        "Outer-loop scripts run QBlade multiple times end-to-end; opt in "
        "explicitly with RUN_QBLADE_OUTER_LOOP=1 (slow).",
    )
    def test_outer_loop_scripts(self):
        for ks, s in enumerate(qblade_outer_loop_scripts):
            with self.subTest(f"Running: {s}", i=ks):
                try:
                    execute_script(s, examples_root="qb_examples")
                    self.assertTrue(True)
                except Exception:
                    self.assertEqual(s, "Success")

    def test_regression_values(self):
        for script, output_names in qblade_regression_scripts.items():
            with self.subTest(f"Regression check: {script}"):
                mod = execute_script(script, examples_root="qb_examples")
                values = _extract_regression_values(mod, output_names)
                compare_regression_values(
                    [values],
                    script.replace("/", "_") + "_regression_values.pkl",
                    directory=this_dir,
                    tol=1e-4,
                )


if __name__ == "__main__":
    if "--train" in sys.argv:
        if not _qblade_dll_available():
            raise RuntimeError(
                "Set QBLADE_DLL_PATH before training regression values."
            )
        for script, output_names in qblade_regression_scripts.items():
            mod = execute_script(script, examples_root="qb_examples")
            values = _extract_regression_values(mod, output_names)
            compare_regression_values(
                [values],
                script.replace("/", "_") + "_regression_values.pkl",
                directory=this_dir,
                train=True,
            )
    else:
        unittest.main()
