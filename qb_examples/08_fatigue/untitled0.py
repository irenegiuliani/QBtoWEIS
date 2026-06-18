"""
Plot selected variables/constraints from an OpenMDAO/WEIS/QBtoWEIS .sql recorder file.

Works for:
- scalar variables: one value per optimization iteration;
- vector variables: one vector per optimization iteration, e.g. tower thickness along height.

Typical usage from terminal:
    python plot_qbtoweis_sql.py --db cases.sql --vars towerse.tower_mass towerse_post.constr_stress

Or edit DB_PATH and USER_VARIABLES below and run from Spyder.
"""

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openmdao.api as om


# =============================================================================
# USER SETTINGS FOR SPYDER / DIRECT RUN
# =============================================================================

DB_PATH = r"C:\Users\irene\NREL_tools\QBtoWEIS\qb_examples\08_fatigue\opt_output\log_iea22mw_umaine.sql"

USER_VARIABLES = [
    # Examples, change these with the variables you want:
    "towerse.tower_mass",
    "towerse_post.constr_stress",
    "towerse_post.constr_shell_buckling",
    "towerse_post.constr_global_buckling",
    "tower_fatigue_post.constr_fatigue",
    "tower_fatigue_post.damage_25y",
    "towerse.t_full",
    "towerse.outer_diameter_full",
]

SOURCE = "driver"          # Usually "driver" for optimization iterations
OUT_DIR = "sql_plots"
SHOW_FIGURES = True
SAVE_CSV = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_filename(name: str) -> str:
    """Convert an OpenMDAO variable name into a safe filename."""
    name = name.replace(".", "__").replace(":", "__").replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name)


def as_1d_array(value):
    """Convert scalar/list/array OpenMDAO values to a 1D numpy array."""
    arr = np.asarray(value)

    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except Exception:
            pass

    arr = np.squeeze(arr)

    if arr.ndim == 0:
        return np.array([float(arr)])

    return np.ravel(arr)


def get_available_sources(case_reader):
    """Return available recorder sources."""
    try:
        return case_reader.list_sources()
    except Exception:
        return []


def get_cases(case_reader, source="driver"):
    """Load cases from a recorder source."""
    try:
        cases = case_reader.get_cases(source, recurse=False)
    except Exception as exc:
        available = get_available_sources(case_reader)
        raise RuntimeError(
            f"Could not read cases from source '{source}'. "
            f"Available sources are: {available}"
        ) from exc

    if len(cases) == 0:
        raise RuntimeError(f"No cases found for source '{source}'.")

    return cases


def get_case_iteration_numbers(cases):
    """
    Return iteration numbers.

    Uses simple 0,1,2,... indexing because this is robust across different
    OpenMDAO recorder naming conventions.
    """
    return np.arange(len(cases), dtype=int)


def try_get_value(case, var_name):
    """
    Try to retrieve a variable from an OpenMDAO Case.

    First uses case.get_val(var_name), then falls back to case.outputs.
    """
    try:
        return case.get_val(var_name)
    except Exception:
        pass

    try:
        if var_name in case.outputs:
            return case.outputs[var_name]
    except Exception:
        pass

    try:
        if var_name in case.inputs:
            return case.inputs[var_name]
    except Exception:
        pass

    raise KeyError(var_name)


def extract_variable_history(cases, var_name):
    """
    Extract one variable across all recorded cases.

    Returns
    -------
    values : list of ndarray
        One 1D array per iteration.
    valid_iterations : ndarray
        Iteration indices where the variable was found.
    """
    values = []
    valid_iterations = []

    for k, case in enumerate(cases):
        try:
            value = try_get_value(case, var_name)
            arr = as_1d_array(value)
        except KeyError:
            continue
        except Exception as exc:
            print(f"[WARNING] Could not read {var_name} at iteration {k}: {exc}")
            continue

        values.append(arr)
        valid_iterations.append(k)

    if len(values) == 0:
        raise KeyError(
            f"Variable '{var_name}' was not found in any recorded case."
        )

    return values, np.asarray(valid_iterations, dtype=int)


def save_scalar_csv(var_name, iterations, scalar_values, out_dir):
    """Save scalar history to CSV."""
    df = pd.DataFrame({
        "iteration": iterations,
        var_name: scalar_values,
    })
    csv_path = out_dir / f"{safe_filename(var_name)}__scalar_history.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def save_vector_csv(var_name, iterations, matrix, out_dir):
    """Save vector history matrix to CSV."""
    columns = [f"{var_name}[{i}]" for i in range(matrix.shape[1])]
    df = pd.DataFrame(matrix, columns=columns)
    df.insert(0, "iteration", iterations)

    csv_path = out_dir / f"{safe_filename(var_name)}__vector_history.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_scalar_history(var_name, iterations, scalar_values, out_dir, show=True):
    """Plot scalar variable vs optimization iteration."""
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    ax.plot(iterations, scalar_values, marker="o", linewidth=1.5)
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel(var_name)
    ax.set_title(f"{var_name} vs iteration")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = out_dir / f"{safe_filename(var_name)}__scalar_history.png"
    fig.savefig(path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return path


def plot_vector_heatmap(var_name, iterations, matrix, out_dir, show=True):
    """
    Plot vector variable as heatmap.

    Matrix shape:
        n_iterations x n_vector_entries
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )

    ax.set_xlabel("Vector index")
    ax.set_ylabel("Optimization iteration index")
    ax.set_title(f"{var_name}: vector evolution")

    if len(iterations) > 1:
        tick_positions = np.linspace(0, len(iterations) - 1, min(8, len(iterations)), dtype=int)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(iterations[tick_positions])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(var_name)

    fig.tight_layout()

    path = out_dir / f"{safe_filename(var_name)}__vector_heatmap.png"
    fig.savefig(path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return path


def plot_vector_final_profile(var_name, matrix, out_dir, show=True):
    """Plot final vector profile at the last available iteration."""
    final_vector = matrix[-1, :]
    x = np.arange(final_vector.size)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    ax.plot(x, final_vector, marker="o", linewidth=1.5)
    ax.set_xlabel("Vector index")
    ax.set_ylabel(var_name)
    ax.set_title(f"{var_name}: final profile")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    path = out_dir / f"{safe_filename(var_name)}__final_profile.png"
    fig.savefig(path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return path


def plot_vector_selected_iterations(var_name, iterations, matrix, out_dir, show=True):
    """
    Plot vector profiles for selected iterations.

    Useful for tower thickness/diameter distributions.
    """
    n_it = matrix.shape[0]
    selected = np.unique(
        np.round(np.linspace(0, n_it - 1, min(6, n_it))).astype(int)
    )

    x = np.arange(matrix.shape[1])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for idx in selected:
        ax.plot(
            x,
            matrix[idx, :],
            marker="o",
            linewidth=1.2,
            label=f"it {iterations[idx]}",
        )

    ax.set_xlabel("Vector index")
    ax.set_ylabel(var_name)
    ax.set_title(f"{var_name}: selected iteration profiles")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    path = out_dir / f"{safe_filename(var_name)}__selected_profiles.png"
    fig.savefig(path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return path


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def process_variable(cases, var_name, out_dir, show=True, save_csv=True):
    """Extract and plot one selected variable."""
    values, valid_iterations = extract_variable_history(cases, var_name)

    sizes = np.array([v.size for v in values], dtype=int)

    if not np.all(sizes == sizes[0]):
        print(
            f"[WARNING] Variable '{var_name}' changes size across iterations. "
            "Only iterations with the most common size will be used."
        )

        unique_sizes, counts = np.unique(sizes, return_counts=True)
        target_size = unique_sizes[np.argmax(counts)]
        keep = sizes == target_size

        values = [v for v, flag in zip(values, keep) if flag]
        valid_iterations = valid_iterations[keep]

    n_comp = values[0].size

    print(f"\nVariable: {var_name}")
    print(f"  iterations found: {len(values)}")
    print(f"  components:       {n_comp}")

    created_files = []

    if n_comp == 1:
        scalar_values = np.array([v[0] for v in values], dtype=float)

        fig_path = plot_scalar_history(
            var_name,
            valid_iterations,
            scalar_values,
            out_dir,
            show=show,
        )
        created_files.append(fig_path)

        if save_csv:
            csv_path = save_scalar_csv(var_name, valid_iterations, scalar_values, out_dir)
            created_files.append(csv_path)

    else:
        matrix = np.vstack(values).astype(float)

        heatmap_path = plot_vector_heatmap(
            var_name,
            valid_iterations,
            matrix,
            out_dir,
            show=show,
        )
        final_path = plot_vector_final_profile(
            var_name,
            matrix,
            out_dir,
            show=show,
        )
        selected_path = plot_vector_selected_iterations(
            var_name,
            valid_iterations,
            matrix,
            out_dir,
            show=show,
        )

        created_files.extend([heatmap_path, final_path, selected_path])

        if save_csv:
            csv_path = save_vector_csv(var_name, valid_iterations, matrix, out_dir)
            created_files.append(csv_path)

    for path in created_files:
        print(f"  saved: {path}")

    return created_files


def list_variables(cases, max_items=200):
    """
    Print available variables from the first case.

    This helps identify exact variable names recorded in the SQL file.
    """
    first_case = cases[0]

    names = []

    try:
        names.extend(list(first_case.outputs.keys()))
    except Exception:
        pass

    try:
        names.extend(list(first_case.inputs.keys()))
    except Exception:
        pass

    names = sorted(set(names))

    print("\nAvailable variables in first recorded case:")
    for name in names[:max_items]:
        print(f"  {name}")

    if len(names) > max_items:
        print(f"\n... printed first {max_items} of {len(names)} variables.")

    print("\nTip: copy the exact variable names into USER_VARIABLES or pass them with --vars.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot selected variables from OpenMDAO/WEIS/QBtoWEIS SQL recorder file."
    )

    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to OpenMDAO SQL recorder file.",
    )

    parser.add_argument(
        "--vars",
        nargs="+",
        default=None,
        help="Variable names to plot.",
    )

    parser.add_argument(
        "--source",
        type=str,
        default=SOURCE,
        help="OpenMDAO recorder source, usually 'driver'.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=OUT_DIR,
        help="Output directory for plots and CSV files.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available variables in the first case and exit.",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show figures interactively, only save them.",
    )

    args = parser.parse_args()

    db_path = Path(args.db if args.db is not None else DB_PATH)
    variables = args.vars if args.vars is not None else USER_VARIABLES

    if not db_path.exists():
        raise FileNotFoundError(f"SQL recorder file not found: {db_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading SQL recorder file: {db_path}")
    print(f"Recorder source:           {args.source}")

    cr = om.CaseReader(str(db_path))
    cases = get_cases(cr, source=args.source)

    print(f"Number of cases loaded:    {len(cases)}")

    if args.list:
        list_variables(cases)
        return

    if not variables:
        print("\nNo variables requested.")
        print("Use --list to inspect available variables, then pass --vars.")
        print("Example:")
        print("  python plot_qbtoweis_sql.py --db cases.sql --list")
        print("  python plot_qbtoweis_sql.py --db cases.sql --vars towerse.tower_mass")
        return

    all_created = []

    for var_name in variables:
        try:
            created = process_variable(
                cases,
                var_name,
                out_dir,
                show=not args.no_show and SHOW_FIGURES,
                save_csv=SAVE_CSV,
            )
            all_created.extend(created)
        except KeyError as exc:
            print(f"\n[WARNING] {exc}")
        except Exception as exc:
            print(f"\n[ERROR] Could not process variable '{var_name}': {exc}")

    print("\nDone.")
    print(f"Generated {len(all_created)} files in: {out_dir}")


if __name__ == "__main__":
    main()