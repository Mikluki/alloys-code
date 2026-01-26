"""
EOS analysis pipeline: normalize and compare VASP vs GNN equation-of-state data.

Workflow:
1. Load VASP and GNN CSVs
2. Merge on (vasp_dir_name, volume_factor) to fill GNN V column from VASP
3. Extract structure IDs from vasp_dir_name
4. For each structure:
   - Compute reference values (V0, E0 per method, B) from VASP points
   - Build normalized EOS and pressure dataframe
   - Generate comparison plots
5. Output derived CSV and summary statistics
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from vsf.core.plot.utils import CHILL_COLORS, save_plot_to_file, set_plot_defaults
from vsf.logging import setup_logging

# Setup logging once
LOGGER = setup_logging(console_level=logging.DEBUG, file_level=logging.DEBUG)

# Setup plotting
set_plot_defaults()

# Color mapping
COLORS = {
    "vasp": CHILL_COLORS.dark_blue,
    "gnn": CHILL_COLORS.orange,
}


def load_both_csvs(
    vasp_path: Path, gnn_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load VASP and GNN EOS CSVs.

    Parameters
    ----------
    vasp_path : Path
        Path to vasp_eos.csv
    gnn_path : Path
        Path to gnn_*.csv

    Returns
    -------
    tuple[DataFrame, DataFrame]
        (df_vasp, df_gnn)
    """
    df_vasp = pd.read_csv(vasp_path)
    df_gnn = pd.read_csv(gnn_path)

    # Ensure numeric columns are floats for safe comparison
    for df in [df_vasp, df_gnn]:
        df["volume_factor"] = pd.to_numeric(df["volume_factor"], errors="coerce")

    LOGGER.info(f"Loaded VASP: {len(df_vasp)} rows from {vasp_path.name}")
    LOGGER.info(f"Loaded GNN: {len(df_gnn)} rows from {gnn_path.name}")

    return df_vasp, df_gnn


def merge_datasets(df_vasp: pd.DataFrame, df_gnn: pd.DataFrame) -> pd.DataFrame:
    """Merge VASP and GNN data, keeping both VASP and GNN rows.

    Uses outer join to preserve VASP rows (especially at volume_factor=1.0 used for references).
    Fills V column for GNN rows from VASP where possible.
    Drops stress_6 and stress_9 columns.

    Parameters
    ----------
    df_vasp : DataFrame
        VASP EOS data
    df_gnn : DataFrame
        GNN EOS data

    Returns
    -------
    DataFrame
        Merged dataset with both VASP and GNN rows
    """
    # Extract V from VASP for imputation
    df_v_map = df_vasp[["vasp_dir_name", "volume_factor", "V"]].rename(
        columns={"V": "V_from_vasp"}
    )

    # Outer join GNN with V from VASP (keeps all rows from both)
    df_gnn_filled = df_gnn.merge(
        df_v_map, on=["vasp_dir_name", "volume_factor"], how="left"
    )

    # Fill empty V in GNN rows from VASP
    missing_v = df_gnn_filled[
        df_gnn_filled["V"].isna() & df_gnn_filled["V_from_vasp"].notna()
    ]
    if len(missing_v) > 0:
        LOGGER.info(f"Filling {len(missing_v)} missing V values for GNN rows from VASP")
        df_gnn_filled.loc[df_gnn_filled["V"].isna(), "V"] = df_gnn_filled.loc[
            df_gnn_filled["V"].isna(), "V_from_vasp"
        ]

    # Drop helper column
    df_gnn_filled = df_gnn_filled.drop(columns=["V_from_vasp"], errors="ignore")

    # Concatenate VASP and GNN rows to keep both
    df_merged = pd.concat([df_vasp, df_gnn_filled], ignore_index=True)

    # Drop stress columns
    df_merged = df_merged.drop(columns=["stress_6", "stress_9"], errors="ignore")

    # Ensure numeric types
    for col in ["V", "E", "P_hydro_GPa"]:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")

    LOGGER.info(
        f"Merged dataset: {len(df_merged)} rows (VASP: {len(df_vasp)}, GNN: {len(df_gnn_filled)})"
    )
    return df_merged


def extract_structure_id(vasp_dir_name: str) -> str:
    """Extract structure ID from vasp_dir_name.

    Example: "TiAl2_mp-567705_0.8" -> "TiAl2_mp-567705"

    Parameters
    ----------
    vasp_dir_name : str
        Directory name from VASP calculation

    Returns
    -------
    str
        Structure ID without volume_factor suffix
    """
    parts = vasp_dir_name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return vasp_dir_name


def get_all_structure_ids(df: pd.DataFrame) -> list[str]:
    """Get unique structure IDs from dataframe.

    Parameters
    ----------
    df : DataFrame
        Merged dataset

    Returns
    -------
    list[str]
        Sorted list of unique structure IDs
    """
    structure_ids = set()
    for vdir in df["vasp_dir_name"].unique():
        sid = extract_structure_id(vdir)
        structure_ids.add(sid)

    return sorted(structure_ids)


def estimate_bulk_modulus_quadratic(
    v_array: np.ndarray, e_array: np.ndarray, v0_idx: int
) -> float:
    """Estimate bulk modulus from local quadratic fit around minimum.

    Fits quadratic E(V) = a*V^2 + b*V + c to 3-5 points around the minimum,
    then computes B = V0 * d2E/dV2 in eV/Ų.

    Parameters
    ----------
    v_array : ndarray
        Volume array (ascending order)
    e_array : ndarray
        Energy array
    v0_idx : int
        Index of minimum energy

    Returns
    -------
    float
        Bulk modulus in eV/Ų
    """
    # Select 3-5 points around minimum
    n = len(v_array)
    start = max(0, v0_idx - 2)
    end = min(n, v0_idx + 3)

    v_fit = v_array[start:end]
    e_fit = e_array[start:end]

    # Quadratic fit: E = a*V^2 + b*V + c
    coeffs = np.polyfit(v_fit, e_fit, 2)
    a = coeffs[0]  # d2E/dV2

    v0 = v_array[v0_idx]
    b_bulk = v0 * a  # in eV/Ų

    return b_bulk


def compute_reference_values(
    df: pd.DataFrame,
    structure_id: str,
) -> dict[str, Any]:
    """Compute reference values (V0, E0, B) for a structure.

    Uses VASP data only:
    - V0_ref: volume at volume_factor==1.0
    - E0_per_method: minimum energy per method
    - B_ref: bulk modulus from quadratic fit on VASP points

    Parameters
    ----------
    df : DataFrame
        Merged dataset
    structure_id : str
        Structure identifier

    Returns
    -------
    dict
        Keys: V0_ref, E0_per_method, B_ref, n_points_per_method
    """
    # Filter to this structure (all volume factors)
    df_struct = df[df["vasp_dir_name"].str.startswith(structure_id + "_")]

    LOGGER.debug(f"Structure ID: {structure_id}")
    LOGGER.debug(
        f"Unique vasp_dir_names: {df_struct['vasp_dir_name'].unique().tolist()}"
    )

    # V0: from VASP at volume_factor==1.0
    vasp_rows = df_struct[df_struct["method"] == "vasp"]
    LOGGER.debug(f"VASP rows found: {len(vasp_rows)}")
    if len(vasp_rows) > 0:
        LOGGER.debug(
            f"VASP volume_factors: {vasp_rows['volume_factor'].unique().tolist()}"
        )
        LOGGER.debug(f"VASP volume_factor dtypes: {vasp_rows['volume_factor'].dtype}")

    v0_rows = df_struct[
        (df_struct["method"] == "vasp") & (df_struct["volume_factor"] == 1.0)
    ]
    LOGGER.debug(f"V0 reference rows (volume_factor==1.0): {len(v0_rows)}")

    if len(v0_rows) == 0:
        LOGGER.warning(f"No VASP reference (volume_factor==1.0) for {structure_id}")
        v0_ref = None
    else:
        v0_ref = v0_rows.iloc[0]["V"]

    # E0: minimum energy per method
    e0_per_method = {}
    n_points_per_method = {}

    for method in df_struct["method"].unique():
        mask = (
            (df_struct["method"] == method)
            & (df_struct["status"] == "ok")
            & (df_struct["V"].notna())
            & (df_struct["E"].notna())
        )
        if mask.any():
            e0_per_method[method] = df_struct[mask]["E"].min()
            n_points_per_method[method] = mask.sum()
        else:
            e0_per_method[method] = None
            n_points_per_method[method] = 0

    # B_ref: bulk modulus from VASP points only
    vasp_mask = (
        (df_struct["method"] == "vasp")
        & (df_struct["status"] == "ok")
        & (df_struct["V"].notna())
        & (df_struct["E"].notna())
    )
    if vasp_mask.sum() >= 3:
        df_vasp_struct = df_struct[vasp_mask].sort_values("V")
        v_arr = df_vasp_struct["V"].values
        e_arr = df_vasp_struct["E"].values
        v0_idx = np.argmin(e_arr)
        b_ref = estimate_bulk_modulus_quadratic(v_arr, e_arr, v0_idx)
    else:
        LOGGER.warning(f"Insufficient VASP points for bulk modulus fit: {structure_id}")
        b_ref = None

    return {
        "V0_ref": v0_ref,
        "E0_per_method": e0_per_method,
        "B_ref": b_ref,
        "n_points_per_method": n_points_per_method,
    }


def build_plot_ready_df(
    df: pd.DataFrame,
    structure_id: str,
    refs: dict[str, Any],
) -> pd.DataFrame:
    """Build normalized dataframe for plotting.

    Adds columns:
    - x_vfrac: V / V0_ref (volume fraction)
    - y_eos: (E - E0) / (B_ref * V0_ref) (normalized energy)
    - x_vol: V (volume for pressure plot)
    - y_p: P_hydro_GPa (pressure)
    - is_eos_valid: mask for EOS plot
    - is_pressure_valid: mask for pressure plot

    Parameters
    ----------
    df : DataFrame
        Merged dataset
    structure_id : str
        Structure identifier
    refs : dict
        Reference values from compute_reference_values

    Returns
    -------
    DataFrame
        Plot-ready dataframe for this structure
    """
    df_struct = df[df["vasp_dir_name"].str.startswith(structure_id + "_")].copy()

    v0 = refs["V0_ref"]
    e0_map = refs["E0_per_method"]
    b_ref = refs["B_ref"]

    # Normalized volume and energy
    if v0 is not None and v0 > 0:
        df_struct["x_vfrac"] = df_struct["V"] / v0
    else:
        df_struct["x_vfrac"] = np.nan

    # Normalized energy per method
    df_struct["y_eos"] = np.nan
    for method in df_struct["method"].unique():
        e0 = e0_map.get(method)
        if e0 is not None and b_ref is not None and v0 is not None and v0 > 0:
            mask = df_struct["method"] == method
            df_struct.loc[mask, "y_eos"] = (df_struct.loc[mask, "E"] - e0) / (
                b_ref * v0
            )

    # Pressure data (already in GPa)
    df_struct["x_vol"] = df_struct["V"]
    df_struct["y_p"] = df_struct["P_hydro_GPa"]

    # Validity masks
    df_struct["is_eos_valid"] = (
        (df_struct["status"] == "ok")
        & (df_struct["V"].notna())
        & (df_struct["E"].notna())
        & (df_struct["x_vfrac"].notna())
        & (df_struct["y_eos"].notna())
    )

    df_struct["is_pressure_valid"] = (
        (df_struct["status"] == "ok")
        & (df_struct["V"].notna())
        & (df_struct["P_hydro_GPa"].notna())
    )

    return df_struct


def plot_eos(
    df: pd.DataFrame,
    structure_id: str,
    refs: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot normalized EOS curves.

    Parameters
    ----------
    df : DataFrame
        Plot-ready dataframe
    structure_id : str
        Structure identifier
    refs : dict
        Reference values
    output_dir : Path
        Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in df["method"].unique():
        mask = (df["method"] == method) & (df["is_eos_valid"])
        if mask.any():
            plot_data = df[mask].sort_values("x_vfrac")
            color = COLORS.get(method, CHILL_COLORS.green)
            ax.plot(
                plot_data["x_vfrac"],
                plot_data["y_eos"],
                "o-",
                label=method,
                color=color,
                markersize=5,
                linewidth=1.5,
                alpha=0.8,
            )

    ax.set_xlabel("Volume Fraction (V/V₀)", fontsize=11)
    ax.set_ylabel("Normalized Energy (eV/Ų)", fontsize=11)
    ax.set_title(f"EOS: {structure_id}", fontsize=12)
    ax.legend(frameon=True, loc="best")
    ax.grid(True, alpha=0.3)

    save_path = output_dir / f"{structure_id}_eos.png"
    save_plot_to_file(fig, save_path, dpi=300)
    LOGGER.info(f"Saved EOS plot: {save_path}")


def plot_pressure(
    df: pd.DataFrame,
    structure_id: str,
    refs: dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot pressure vs volume.

    Parameters
    ----------
    df : DataFrame
        Plot-ready dataframe
    structure_id : str
        Structure identifier
    refs : dict
        Reference values
    output_dir : Path
        Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in df["method"].unique():
        mask = (df["method"] == method) & (df["is_pressure_valid"])
        if mask.any():
            plot_data = df[mask].sort_values("x_vol")
            color = COLORS.get(method, CHILL_COLORS.green)
            ax.plot(
                plot_data["x_vol"],
                plot_data["y_p"],
                "o-",
                label=method,
                color=color,
                markersize=5,
                linewidth=1.5,
                alpha=0.8,
            )

    ax.set_xlabel("Volume (Ų)", fontsize=11)
    ax.set_ylabel("Hydrostatic Pressure (GPa)", fontsize=11)
    ax.set_title(f"Pressure: {structure_id}", fontsize=12)
    ax.legend(frameon=True, loc="best")
    ax.grid(True, alpha=0.3)

    save_path = output_dir / f"{structure_id}_pressure.png"
    save_plot_to_file(fig, save_path, dpi=300)
    LOGGER.info(f"Saved pressure plot: {save_path}")


def write_derived_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Write plot-ready dataframe to CSV.

    Parameters
    ----------
    df : DataFrame
        Complete plot-ready dataset
    output_path : Path
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select relevant columns for export
    cols = [
        "method",
        "vasp_dir_name",
        "volume_factor",
        "status",
        "V",
        "E",
        "P_hydro_GPa",
        "x_vfrac",
        "y_eos",
        "x_vol",
        "y_p",
        "is_eos_valid",
        "is_pressure_valid",
    ]
    cols = [c for c in cols if c in df.columns]

    df[cols].to_csv(output_path, index=False)
    LOGGER.info(f"Saved derived CSV: {output_path}")


def write_summary_csv(
    all_refs: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Write summary statistics (reference values, bulk moduli).

    Parameters
    ----------
    all_refs : dict
        Keys: structure_id, values: refs dict from compute_reference_values
    output_path : Path
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid in sorted(all_refs.keys()):
        refs = all_refs[sid]
        row = {
            "structure_id": sid,
            "V0_ref": refs["V0_ref"],
            "B_ref_eV_A3": refs["B_ref"],
        }

        # Add E0 and n_points per method
        for method in refs["E0_per_method"].keys():
            row[f"E0_{method}"] = refs["E0_per_method"].get(method)
            row[f"n_points_{method}"] = refs["n_points_per_method"].get(method, 0)

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_path, index=False)
    LOGGER.info(f"Saved summary CSV: {output_path}")


def main(
    vasp_csv: Path,
    gnn_csv: Path,
    output_dir: Path = None,
) -> None:
    """Main workflow: load, merge, compute, plot, export.

    Parameters
    ----------
    vasp_csv : Path
        Path to vasp_eos.csv
    gnn_csv : Path
        Path to gnn_*_eos.csv
    output_dir : Path, optional
        Output directory (default: current directory)
    """
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Output directory: {output_dir}")

    # Load and merge
    df_vasp, df_gnn = load_both_csvs(Path(vasp_csv), Path(gnn_csv))
    df_merged = merge_datasets(df_vasp, df_gnn)

    # Process each structure
    structure_ids = get_all_structure_ids(df_merged)
    LOGGER.info(f"Found {len(structure_ids)} structures")

    all_refs = {}
    all_plot_dfs = []

    for sid in structure_ids:
        LOGGER.info(f"Processing {sid}...")

        # Compute references
        refs = compute_reference_values(df_merged, sid)
        all_refs[sid] = refs

        # Build plot-ready data
        df_plot = build_plot_ready_df(df_merged, sid, refs)
        all_plot_dfs.append(df_plot)

        # Generate plots
        if refs["V0_ref"] is not None and refs["B_ref"] is not None:
            plot_eos(df_plot, sid, refs, plots_dir)
            plot_pressure(df_plot, sid, refs, plots_dir)
        else:
            LOGGER.warning(f"Skipping plots for {sid}: insufficient reference values")

    # Write outputs
    df_all_plots = pd.concat(all_plot_dfs, ignore_index=True)
    write_derived_csv(df_all_plots, output_dir / "derived_eos.csv")
    write_summary_csv(all_refs, output_dir / "summary.csv")

    LOGGER.info("Complete!")


if __name__ == "__main__":
    # Example usage
    main(
        vasp_csv=Path("20-results/vasp_eos.csv"),
        gnn_csv=Path("20-results/gnn_Mace_mpa_0_eos.csv"),
        output_dir=Path("20-results/outputs"),
    )
