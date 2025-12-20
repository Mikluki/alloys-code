"""
Decorrelate VASP MD trajectories using fraction-moved metric + pymbar/alchemlyb.

Algorithm:
1. Detect burn-in from potential energy (pymbar.detect_equilibration)
2. Build structural time series: fraction of atoms moved ≥ δ Å between frames
3. Subsample post-burn-in frames using alchemlyb.statistical_inefficiency
4. Generate diagnostic plots for each trajectory

Dependencies: ase, numpy, pandas, matplotlib, pymbar, alchemlyb
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alchemlyb.preprocessing.subsampling import statistical_inefficiency as alch_statinf
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic
from pymbar.timeseries import detect_equilibration
from pymbar.timeseries import statistical_inefficiency as g_est
from pymbar.utils import ParameterError

from vsf.liquid.extract import ConfigurationData

LOGGER = logging.getLogger(__name__)
DPI = 300


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class TrajectoryDecorrelationResult:
    """Results from decorrelating one MD trajectory."""

    source_dir: Path
    element: str
    total_frames: int
    burn_in_frame: int
    selected_frames: np.ndarray  # Frame indices in original trajectory
    decorrelated_configs: List[ConfigurationData]
    g_energy: float
    g_struct: float
    Neff_energy: float
    frame_lag: int
    delta: float
    has_energy: bool

    @property
    def n_selected(self) -> int:
        return len(self.selected_frames)

    @property
    def sampling_efficiency(self) -> float:
        return self.n_selected / self.total_frames if self.total_frames > 0 else 0.0

    @property
    def post_burnin_frames(self) -> int:
        return self.total_frames - self.burn_in_frame


@dataclass
class ElementDecorrelationResults:
    """Aggregated results for one element across all MD runs."""

    element: str
    trajectory_results: List[TrajectoryDecorrelationResult]
    total_configs_available: int
    total_configs_selected: int

    @property
    def n_trajectories(self) -> int:
        return len(self.trajectory_results)

    @property
    def sampling_efficiency(self) -> float:
        if self.total_configs_available == 0:
            return 0.0
        return self.total_configs_selected / self.total_configs_available

    def get_all_decorrelated_configs(self) -> List[ConfigurationData]:
        """Get all decorrelated configs across all trajectories."""
        configs = []
        for traj_result in self.trajectory_results:
            configs.extend(traj_result.decorrelated_configs)
        return configs


# ============================================================================
# CONVERSION UTILITIES
# ============================================================================


def config_to_atoms(config: ConfigurationData) -> Atoms:
    """
    Convert ConfigurationData to ASE Atoms with energy.

    Args:
        config: Configuration to convert

    Returns:
        ASE Atoms object with energy attached via calculator
    """
    structure = config.structure

    atoms = Atoms(
        symbols=[str(s) for s in structure.species],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )

    # Attach energy if available
    if config.energy_sigma_to_0 is not None:
        calc = SinglePointCalculator(atoms, energy=config.energy_sigma_to_0)
        atoms.calc = calc

    return atoms


# ============================================================================
# CORE DECORRELATION ALGORITHMS (adapted from external API)
# ============================================================================


def _mic_disp(R2: np.ndarray, R1: np.ndarray, cell, pbc) -> np.ndarray:
    """
    Return minimum-image displacement R2-R1 for given cell and PBC.

    Args:
        R2, R1: (N,3) arrays of Cartesian positions (Å)
        cell: (3,3) array-like cell vectors (Å)
        pbc: (3,) bool periodic boundary flags

    Returns:
        (N,3) array of minimum-image displacements (Å)
    """
    res = find_mic(R2 - R1, cell, pbc)
    return res[0] if isinstance(res, tuple) else res


def extract_energy_series(atoms: List[Atoms]) -> Optional[pd.Series]:
    """
    Extract potential energy per frame (eV) if available.

    Args:
        atoms: List of ASE Atoms with calculators

    Returns:
        pandas.Series indexed by frame, or None if energies unavailable
    """
    try:
        E = [a.get_potential_energy(apply_constraint=False) for a in atoms]
        return pd.Series(E, index=pd.Index(range(len(E)), name="time"), name="E_pot_eV")
    except Exception:
        return None


def compute_fraction_moved_series(
    atoms: List[Atoms], frame_lag: int, delta: float
) -> pd.Series:
    """
    Compute fraction of atoms with |Δr| ≥ delta between frames t and t−lag.

    Args:
        atoms: List of ASE Atoms
        frame_lag: Frame separation (steps)
        delta: Displacement threshold (Å)

    Returns:
        pandas.Series with values in [0,1], indexed by frame
    """
    vals, idx = [], []
    for t in range(frame_lag, len(atoms)):
        a2, a1 = atoms[t], atoms[t - frame_lag]
        d = _mic_disp(
            a2.get_positions(), a1.get_positions(), a2.get_cell(), a2.get_pbc()
        )
        vals.append((np.linalg.norm(d, axis=1) >= delta).mean())
        idx.append(t)
    return pd.Series(
        vals,
        index=pd.Index(idx, name="time"),
        name=f"FracMoved_k{frame_lag}_d{delta:.4f}",
    )


def subsample_structural(
    series: pd.Series,
    t0: int,
    conservative: bool,
    step_fallback: Optional[int],
) -> Tuple[np.ndarray, float]:
    """
    Subsample frames post-burn-in using alchemlyb.

    Args:
        series: Structural time series (indexed by frame)
        t0: Burn-in start frame (inclusive)
        conservative: If True, integer spacing (fewer frames)
        step_fallback: Stride if statistical inefficiency fails (None = hard fail)

    Returns:
        frames: Selected frame indices (0-based)
        g: Statistical inefficiency of series after t0; NaN if not computed

    Raises:
        ValueError: If statistical inefficiency fails and step_fallback is None
    """
    s_eq = series.loc[series.index >= t0]

    if len(s_eq) < 3:
        raise ValueError(f"Insufficient post-burn-in frames: {len(s_eq)} < 3")

    if float(np.nanstd(s_eq.values)) < 1e-12:
        LOGGER.warning("Structural series has no variance post-burn-in")
        if step_fallback is not None:
            return s_eq.index.values[:: max(1, step_fallback)], np.nan
        else:
            raise ValueError("Structural series has no variance and step_fallback=None")

    df_eq = pd.DataFrame({"frame": s_eq.index.values}, index=s_eq.index)

    try:
        df_sub = alch_statinf(
            df_eq,
            series=s_eq,
            drop_duplicates=True,
            sort=True,
            conservative=conservative,
        )
        frames = df_sub["frame"].to_numpy()
        try:
            g = float(g_est(s_eq.values))
        except Exception:
            g = np.nan
        return frames, g

    except ParameterError as e:
        if step_fallback is not None:
            LOGGER.warning(
                f"Statistical inefficiency estimation failed: {e}. "
                f"Using fallback stride={step_fallback}"
            )
            return s_eq.index.values[:: max(1, step_fallback)], np.nan
        else:
            raise ValueError(
                f"Statistical inefficiency estimation failed: {e}. "
                f"Set step_fallback to use fixed stride."
            ) from e


def naive_acf(x: np.ndarray, max_frame_lag: int = 200) -> np.ndarray:
    """
    Compute simple normalized ACF up to max_frame_lag (NaN-safe).

    Args:
        x: Time series values
        max_frame_lag: Maximum lag to compute

    Returns:
        ACF[0..max_frame_lag]; ACF[0]=1
    """
    x = np.asarray(x, float)
    x -= np.nanmean(x)
    var = np.nanvar(x)
    if var == 0:
        return np.r_[1.0, np.zeros(max_frame_lag)]
    L = min(max_frame_lag, len(x) - 2)
    acf = np.empty(L + 1)
    acf[0] = 1.0
    for k in range(1, L + 1):
        acf[k] = np.nanmean(x[:-k] * x[k:]) / var
    return acf


# ============================================================================
# PLOTTING
# ============================================================================


def plot_trajectory_diagnostics(
    S: pd.Series,
    frames: np.ndarray,
    t0: int,
    output_path: Path,
    title_prefix: str = "",
):
    """
    Generate diagnostic plots for one trajectory.

    Creates three plots:
    1. Time series with burn-in and selected frames
    2. Histogram of post-burn-in values
    3. Autocorrelation function

    Args:
        S: Structural series (fraction moved)
        frames: Selected frame indices
        t0: Burn-in start frame
        output_path: Path to save plot
        title_prefix: Optional title prefix
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Time series
    ax = axes[0]
    ax.plot(S.index, S.values, lw=1.5, label=S.name, alpha=0.7)
    ax.axvline(t0, ls="--", c="red", lw=2, label=f"Burn-in (t₀={t0})")
    ax.scatter(
        frames, S.loc[frames], s=30, c="orange", zorder=3, label="Selected frames"
    )
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Fraction moved", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Structural Time Series", fontsize=12)

    # Plot 2: Histogram
    S_eq = S.loc[S.index >= t0]
    ax = axes[1]
    ax.hist(S_eq.values, bins=30, edgecolor="k", alpha=0.7)
    ax.set_xlabel("Fraction moved", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Post Burn-in Distribution", fontsize=12)
    ax.grid(alpha=0.3, axis="y")

    # Plot 3: ACF
    acf = naive_acf(S_eq.values)
    ax = axes[2]
    ax.plot(np.arange(len(acf)), acf, lw=1.5)
    ax.axhline(1 / np.e, ls="--", c="red", label="1/e threshold")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("Frame Lag", fontsize=11)
    ax.set_ylabel("Autocorrelation", fontsize=11)
    ax.set_title("ACF (Post Burn-in)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    LOGGER.info(f"Diagnostic plot saved: {output_path}")


# ============================================================================
# DECORRELATION ANALYSIS
# ============================================================================


def decorrelate_trajectory(
    configs: List[ConfigurationData],
    frame_lag: int = 10,
    delta: float = 0.005,
    conservative: bool = False,
    step_fallback: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> TrajectoryDecorrelationResult:
    """
    Decorrelate a single MD trajectory.

    All configs must be from the same source_dir (single continuous trajectory).

    Args:
        configs: Configurations from ONE source_dir
        frame_lag: Frame separation for fraction-moved metric
        delta: Displacement threshold (Å)
        conservative: alchemlyb conservative mode (fewer frames)
        step_fallback: Fallback stride if estimation fails (None = hard fail)
        output_dir: Directory to save diagnostic plots (None = no plots)

    Returns:
        TrajectoryDecorrelationResult with decorrelated configs and diagnostics

    Raises:
        ValueError: If configs from multiple sources or insufficient data
    """
    if not configs:
        raise ValueError("No configs provided")

    # Verify all from same source
    source_dirs = {c.source_dir for c in configs}
    if len(source_dirs) > 1:
        raise ValueError(f"Configs from multiple sources: {source_dirs}")

    source_dir = configs[0].source_dir
    element = configs[0].element

    # Sort by timestep
    sorted_configs = sorted(configs, key=lambda x: x.time_step)

    LOGGER.info(f"Processing trajectory: {element}/{source_dir.name}")
    LOGGER.info(f"  Total frames: {len(sorted_configs)}")

    # Convert to ASE Atoms
    atoms = [config_to_atoms(c) for c in sorted_configs]

    # Extract energy series
    E = extract_energy_series(atoms)
    has_energy = E is not None

    if not has_energy:
        raise ValueError(
            f"No energies available for {element}/{source_dir.name}. "
            f"Burn-in detection requires energies. Skipping trajectory."
        )

    # Detect burn-in from energy
    t0, gE, NeffE = detect_equilibration(E.values)
    t0 = int(t0)
    LOGGER.info(f"  Burn-in detected at frame: {t0}")
    LOGGER.info(f"  Energy g: {gE:.2f}, Neff: {NeffE:.1f}")

    # Compute structural time series
    S = compute_fraction_moved_series(atoms, frame_lag, delta)
    LOGGER.info(f"  Structural series computed (frame_lag={frame_lag}, delta={delta}Å)")

    # Subsample post-burn-in frames
    try:
        frames, gS = subsample_structural(S, (t0), conservative, step_fallback)
        LOGGER.info(f"  Subsampling: {len(frames)} frames selected")
        LOGGER.info(f"  Structural g: {gS:.2f}")
    except ValueError as e:
        LOGGER.error(f"  Subsampling failed: {e}")
        raise

    # Select decorrelated configs
    decorrelated = [sorted_configs[i] for i in frames]

    # Generate diagnostic plot
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        plot_path = output_dir / f"decorr_{element}_{source_dir.name}.png"
        plot_trajectory_diagnostics(
            S, frames, t0, plot_path, title_prefix=f"{element} / {source_dir.name}"
        )

    result = TrajectoryDecorrelationResult(
        source_dir=source_dir,
        element=element,
        total_frames=len(sorted_configs),
        burn_in_frame=t0,
        selected_frames=frames,
        decorrelated_configs=decorrelated,
        g_energy=gE,
        g_struct=gS,
        Neff_energy=NeffE,
        frame_lag=frame_lag,
        delta=delta,
        has_energy=has_energy,
    )

    LOGGER.info(
        f"  Result: {result.n_selected}/{result.total_frames} frames "
        f"({result.sampling_efficiency:.1%} efficiency)"
    )

    return result


def analyze_element_decorrelation(
    configs: List[ConfigurationData],
    frame_lag: int = 10,
    delta: float = 0.005,
    conservative: bool = False,
    step_fallback: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> ElementDecorrelationResults:
    """
    Decorrelate all trajectories for one element.

    Groups configs by source_dir and processes each MD run separately.

    Args:
        configs: All configurations for one element
        frame_lag: Frame separation for fraction-moved metric
        delta: Displacement threshold (Å)
        conservative: alchemlyb conservative mode
        step_fallback: Fallback stride if estimation fails (None = hard fail)
        output_dir: Directory to save plots

    Returns:
        ElementDecorrelationResults with aggregated results
    """
    element = configs[0].element

    LOGGER.info("=" * 70)
    LOGGER.info(f"ELEMENT: {element}")
    LOGGER.info("=" * 70)

    # Group by source_dir (each is a separate trajectory)
    by_source = {}
    for config in configs:
        if config.source_dir not in by_source:
            by_source[config.source_dir] = []
        by_source[config.source_dir].append(config)

    LOGGER.info(f"Found {len(by_source)} MD trajectories for {element}")

    # Process each trajectory
    trajectory_results = []
    total_selected = 0

    for source_dir, traj_configs in by_source.items():
        try:
            result = decorrelate_trajectory(
                traj_configs,
                frame_lag=frame_lag,
                delta=delta,
                conservative=conservative,
                step_fallback=step_fallback,
                output_dir=output_dir,
            )
            trajectory_results.append(result)
            total_selected += result.n_selected

        except Exception as e:
            LOGGER.error(f"Failed to decorrelate {source_dir.name}: {e}")
            continue

    return ElementDecorrelationResults(
        element=element,
        trajectory_results=trajectory_results,
        total_configs_available=len(configs),
        total_configs_selected=total_selected,
    )


def analyze_all_elements_decorrelation(
    configs_by_element: Dict[str, List[ConfigurationData]],
    output_dir: Optional[Path] = None,
    frame_lag: int = 10,
    delta: float = 0.005,
    conservative: bool = False,
    step_fallback: Optional[int] = None,
) -> Dict[str, ElementDecorrelationResults]:
    """
    Decorrelate all elements using pymbar burn-in detection + alchemlyb subsampling.

    Args:
        configs_by_element: Dict mapping element -> configs
        output_dir: Directory for plots and reports
        frame_lag: Frame separation for fraction-moved metric (default: 10)
        delta: Displacement threshold in Å (default: 0.005)
        conservative: alchemlyb conservative mode - fewer frames (default: False)
        step_fallback: Fallback stride if estimation fails (None = hard fail)

    Returns:
        Dict mapping element -> ElementDecorrelationResults

    Raises:
        ValueError: If trajectory decorrelation fails and step_fallback is None
    """
    if output_dir is None:
        output_dir = Path("x-decorrelation_analysis")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    LOGGER.info("=" * 70)
    LOGGER.info("DECORRELATION ANALYSIS")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(
        f"Parameters: frame_lag={frame_lag}, delta={delta}Å, conservative={conservative}"
    )
    LOGGER.info(
        f"Fallback stride: {step_fallback if step_fallback else 'None (hard fail)'}"
    )

    results = {}

    for element, configs in configs_by_element.items():
        if len(configs) < 10:
            LOGGER.warning(
                f"Skipping {element}: only {len(configs)} configs (need ≥10)"
            )
            continue

        try:
            result = analyze_element_decorrelation(
                configs,
                frame_lag=frame_lag,
                delta=delta,
                conservative=conservative,
                step_fallback=step_fallback,
                output_dir=output_dir,
            )
            results[element] = result

        except Exception as e:
            LOGGER.error(f"Failed to analyze {element}: {e}")
            continue

    # Generate summary report
    _generate_summary_report(
        results, output_dir, frame_lag, delta, conservative, step_fallback
    )

    LOGGER.info("=" * 70)
    LOGGER.info("ANALYSIS COMPLETE")
    LOGGER.info("=" * 70)
    LOGGER.info(
        f"Successfully analyzed: {len(results)}/{len(configs_by_element)} elements"
    )

    return results


# ============================================================================
# REPORTING
# ============================================================================


def _generate_summary_report(
    results: Dict[str, ElementDecorrelationResults],
    output_dir: Path,
    frame_lag: int,
    delta: float,
    conservative: bool,
    step_fallback: Optional[int],
):
    """Generate summary report with statistics and plot interpretation guide."""
    report_path = output_dir / "decorrelation_summary.txt"

    with open(report_path, "w") as f:
        f.write("DECORRELATION ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"frame_lag (k): {frame_lag} frames\n")
        f.write(f"Delta (δ): {delta} Å\n")
        f.write(f"Conservative mode: {conservative}\n")
        f.write(
            f"Fallback stride: {step_fallback if step_fallback else 'None (hard fail)'}\n\n"
        )

        f.write("RESULTS BY ELEMENT\n")
        f.write("=" * 70 + "\n\n")

        for element, result in results.items():
            f.write(f"{element}:\n")
            f.write(f"  MD trajectories processed: {result.n_trajectories}\n")
            f.write(f"  Total configs available: {result.total_configs_available}\n")
            f.write(f"  Total configs selected: {result.total_configs_selected}\n")
            f.write(f"  Overall efficiency: {result.sampling_efficiency:.1%}\n\n")

            for traj_result in result.trajectory_results:
                f.write(f"  Trajectory: {traj_result.source_dir.name}\n")
                f.write(f"    Total frames: {traj_result.total_frames}\n")
                f.write(f"    Burn-in frame: {traj_result.burn_in_frame}\n")
                f.write(f"    Post burn-in: {traj_result.post_burnin_frames}\n")
                f.write(f"    Selected: {traj_result.n_selected}\n")
                f.write(f"    Efficiency: {traj_result.sampling_efficiency:.1%}\n")
                f.write(f"    g_energy: {traj_result.g_energy:.2f}\n")
                f.write(f"    g_struct: {traj_result.g_struct:.2f}\n")
                f.write(f"    Neff_energy: {traj_result.Neff_energy:.1f}\n")
                f.write(f"\n")

            f.write("\n")

        f.write("DIAGNOSTIC PLOTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            "Each trajectory has a diagnostic plot: decorr_{element}_{source}.png\n\n"
        )

        f.write("HOW TO INTERPRET PLOTS:\n")
        f.write("-" * 70 + "\n\n")

        f.write("LEFT PANEL - Structural Time Series:\n")
        f.write("  • Blue line: Fraction of atoms that moved ≥ δ Å over frame_lag k\n")
        f.write("  • Red dashed line: Burn-in point (t₀) detected from energy\n")
        f.write("  • Orange points: Selected decorrelated frames\n")
        f.write("  • What to look for:\n")
        f.write("    - Equilibration: series should stabilize after burn-in\n")
        f.write("    - Selected frames should span equilibrated region\n\n")

        f.write("MIDDLE PANEL - Distribution:\n")
        f.write("  • Histogram of fraction-moved values after burn-in\n")
        f.write("  • What to look for:\n")
        f.write("    - Should show reasonable spread (not all same value)\n")
        f.write("    - Peak around typical structural fluctuation level\n\n")

        f.write("RIGHT PANEL - Autocorrelation Function:\n")
        f.write("  • ACF of fraction-moved series (post burn-in)\n")
        f.write("  • Red dashed line: 1/e threshold (~0.368)\n")
        f.write("  • What to look for:\n")
        f.write("    - Decay to zero indicates decorrelation\n")
        f.write("    - Fast decay = good (frames become independent quickly)\n")
        f.write("    - Slow decay = poor (need longer trajectories)\n")
        f.write("    - No decay = problem (non-ergodic or insufficient sampling)\n\n")

        f.write("KEY METRICS:\n")
        f.write("-" * 70 + "\n\n")
        f.write("g_energy: Statistical inefficiency from potential energy\n")
        f.write("  • g ≈ 1: uncorrelated data (ideal)\n")
        f.write(
            "  • g > 1: correlated data (g=5 means 5 frames needed per independent sample)\n\n"
        )

        f.write("g_struct: Statistical inefficiency from structural metric\n")
        f.write("  • Same interpretation as g_energy\n")
        f.write("  • Used for actual subsampling\n\n")

        f.write("Neff_energy: Effective number of independent energy samples\n")
        f.write("  • Neff = N_total / g_energy\n")
        f.write("  • Higher is better\n\n")

        f.write("TROUBLESHOOTING:\n")
        f.write("-" * 70 + "\n\n")
        f.write("If few frames selected:\n")
        f.write("  • Check g_struct: high values indicate strong correlation\n")
        f.write("  • Burn-in may be consuming many frames\n")
        f.write("  • Consider longer MD runs or adjust frame_lag/delta parameters\n\n")

        f.write("If analysis fails:\n")
        f.write("  • Check for missing energies (required for burn-in detection)\n")
        f.write("  • Check structural series has variance (not frozen)\n")
        f.write("  • Use step_fallback parameter for problematic trajectories\n")

    LOGGER.info(f"Summary report saved: {report_path}")
