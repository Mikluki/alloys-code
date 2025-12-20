import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from sklearn.metrics import r2_score

from .utils import CHILL_COLORS, save_plot_to_file, set_plot_defaults

LOGGER = logging.getLogger(__name__)


@dataclass
class ScatterPlotConfig:
    """Configuration for energy comparison scatter plots."""

    figsize: tuple[int, int] = (8, 6)
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    dpi: int | None = 100
    show_diagonal: bool = True
    data_type: str = "value"
    x_range: tuple[float, float] | None = None
    y_range: tuple[float, float] | None = None


# ============================================================================
# Extract Functions
# ============================================================================


def extract_scatter_data(
    structures: List[Dict],
    energy_source1: str,
    energy_source2: str,
    data_type: str = "value",
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract energy data for scatter plot from structures.

    Parameters
    ----------
    structures : List[Dict]
        List of structure result dictionaries
    energy_source1 : str
        First energy source identifier (X-axis)
    energy_source2 : str
        Second energy source identifier (Y-axis)
    data_type : str, optional
        Data type to extract ('value' or 'delta'), by default "value"
    x_range : tuple[float, float] | None, optional
        Physical range filter for X-axis (min, max), by default None
    y_range : tuple[float, float] | None, optional
        Physical range filter for Y-axis (min, max), by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two aligned arrays of energies (x_energies, y_energies)

    Raises
    ------
    ValueError
        If data_type is invalid
    """
    if data_type not in ["value", "delta"]:
        raise ValueError(f"data_type must be 'value' or 'delta', got '{data_type}'")

    x_data = []
    y_data = []

    for structure in structures:
        formation_data = structure.get("formation_energy", {})
        results = formation_data.get("results", {})

        if energy_source1 not in results:
            LOGGER.warning(
                f"Energy source '{energy_source1}' not found in {structure.get('name')}"
            )
            continue

        if energy_source2 not in results:
            LOGGER.warning(
                f"Energy source '{energy_source2}' not found in {structure.get('name')}"
            )
            continue

        try:
            energy1 = results[energy_source1][data_type]
            energy2 = results[energy_source2][data_type]

            x_data.append(energy1)
            y_data.append(energy2)

        except (KeyError, TypeError) as e:
            LOGGER.warning(f"Failed to extract energy data: {e}")
            continue

    x_array = np.array(x_data)
    y_array = np.array(y_data)

    total_extracted = len(x_array)

    if x_range is not None:
        x_mask = (x_array >= x_range[0]) & (x_array <= x_range[1])
        x_array = x_array[x_mask]
        y_array = y_array[x_mask]
        LOGGER.info(
            f"X-axis range {x_range}: Kept {len(x_array)} of {total_extracted} points"
        )

    if y_range is not None:
        before_y_filter = len(y_array)
        y_mask = (y_array >= y_range[0]) & (y_array <= y_range[1])
        x_array = x_array[y_mask]
        y_array = y_array[y_mask]
        LOGGER.info(
            f"Y-axis range {y_range}: Kept {len(y_array)} of {before_y_filter} points"
        )

    LOGGER.info(
        f"({energy_source1} vs {energy_source2}): "
        f"Final dataset has {len(x_array)} points (from {total_extracted} extracted)"
    )

    return x_array, y_array


# ============================================================================
# Render Functions
# ============================================================================


def _render_energy_scatter_plot(
    x_array: np.ndarray,
    y_array: np.ndarray,
    energy_source1: str,
    energy_source2: str,
    config: ScatterPlotConfig,
    ax: Axes | None = None,
    fig: Figure | None = None,
    label: str | None = None,
    color: str | None = None,
) -> Tuple[Figure, Axes]:
    """
    Render formation energy scatter plot.

    Parameters
    ----------
    x_array : np.ndarray
        X-axis energy values
    y_array : np.ndarray
        Y-axis energy values
    energy_source1 : str
        First energy source identifier (X-axis label)
    energy_source2 : str
        Second energy source identifier (Y-axis label)
    config : ScatterPlotConfig
        Configuration for plot appearance and styling
    ax : Axes | None, optional
        Existing axes to plot on (for multi-series plots), by default None
    fig : Figure | None, optional
        Existing figure to plot on, by default None
    label : str | None, optional
        Label for legend, by default None
    color : str | None, optional
        Scatter point color (auto-cycles if None), by default None

    Returns
    -------
    Tuple[Figure, Axes]
        Matplotlib figure and axes objects
    """
    LOGGER.info(
        f"Scatter plot comparison between `{energy_source1}` and `{energy_source2}`"
    )

    # Create new figure/axes if not provided
    if fig is None or ax is None:
        font_size, label_size = set_plot_defaults(linewidth_coeff=0.8, title_size=12)
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        # Set labels and title only for new axes
        energy_str = r"E$_\text{formation}$"
        ax.set_xlabel(f"{energy_source1} {energy_str} (eV/atom)")
        ax.set_ylabel(f"GNN {energy_str} (eV/atom)")
        ax.set_title(f"GNN = `{energy_source2}`")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    # Auto-select color by cycling through CHILL_COLORS
    if color is None:
        color_list = list(CHILL_COLORS)
        n_existing = len(ax.collections)
        color = color_list[n_existing % len(color_list)]

    # Plot scatter points
    ax.scatter(
        x_array,
        y_array,
        color=color,
        alpha=0.6,
        s=20,
        edgecolors="none",
        label=label,
    )

    # Add diagonal reference line
    if config.show_diagonal:
        ax_min = min(x_array.min(), y_array.min())
        ax_max = max(x_array.max(), y_array.max())
        ax.plot(
            [ax_min, ax_max],
            [ax_min, ax_max],
            "--",
            alpha=0.8,
            linewidth=1,
            color="#88cff6",
            label=None,
        )

    # Apply limits if provided
    if config.xlim is not None:
        ax.set_xlim(config.xlim)
    if config.ylim is not None:
        ax.set_ylim(config.ylim)

    return fig, ax


# ============================================================================
# Statistics Utilities
# ============================================================================


def _calculate_scatter_stats(
    x_arrays: List[np.ndarray], y_arrays: List[np.ndarray]
) -> Dict[str, float]:
    """
    Calculate combined statistics from multiple datasets.

    Parameters
    ----------
    x_arrays : List[np.ndarray]
        List of X-axis data arrays
    y_arrays : List[np.ndarray]
        List of Y-axis data arrays

    Returns
    -------
    Dict[str, float]
        Dictionary with 'n_points', 'mae', and 'r_squared' keys

    Raises
    ------
    ValueError
        If arrays are empty or mismatched
    """
    if not x_arrays or not y_arrays:
        raise ValueError("Arrays cannot be empty")

    x_combined = np.concatenate(x_arrays)
    y_combined = np.concatenate(y_arrays)

    n_points = len(x_combined)
    mae = float(np.mean(np.abs(y_combined - x_combined)))
    r_squared = r2_score(x_combined, y_combined)

    LOGGER.info(f"Combined statistics: N={n_points}, MAE={mae:.3f}, R²={r_squared:.2f}")

    return {
        "n_points": n_points,
        "mae": mae,
        "r_squared": r_squared,
    }


def add_stats_box(
    ax: Axes,
    x_arrays: List[np.ndarray],
    y_arrays: List[np.ndarray],
    loc: str = "best",
    show_n_points: bool = True,
) -> None:
    """
    Add combined statistics box to existing axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to add stats box to
    x_arrays : List[np.ndarray]
        List of X-axis data arrays
    y_arrays : List[np.ndarray]
        List of Y-axis data arrays
    loc : str, optional
        Legend location, by default "best"
    show_n_points : bool, optional
        Include number of points in stats box, by default True
    """
    stats = _calculate_scatter_stats(x_arrays, y_arrays)

    n_points_str = f"N = {stats['n_points']:,}\n" if show_n_points else ""
    stats_text = (
        f"{n_points_str}MAE = {stats['mae']:.3f}\nR² = {stats['r_squared']:.2f}"
    )

    # Get existing handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Add stats box as legend entry
    handles.append(Patch(alpha=0))
    labels.append(stats_text)

    ax.legend(
        handles=handles,
        labels=labels,
        loc=loc,
    )


# ============================================================================
# Orchestration Functions
# ============================================================================


def plot_energy_comparison_scatter(
    structures: List[Dict],
    energy_source1: str,
    energy_source2: str,
    config: ScatterPlotConfig | None = None,
    save_path: Path | None = None,
    label: str | None = None,
    color: str | None = None,
) -> Figure:
    """
    Plot formation energy scatter comparing two energy sources.

    Parameters
    ----------
    structures : List[Dict]
        List of structure result dictionaries
    energy_source1 : str
        First energy source identifier (X-axis)
    energy_source2 : str
        Second energy source identifier (Y-axis)
    config : ScatterPlotConfig, optional
        Plot configuration (uses defaults if None)
    save_path : Path | None, optional
        Path to save figure, by default None
    label : str | None, optional
        Label for legend, by default None
    color : str | None, optional
        Scatter point color (auto-cycles if None), by default None

    Returns
    -------
    Figure
        Matplotlib figure object

    Raises
    ------
    ValueError
        If no valid energy pairs are found
    """
    if config is None:
        config = ScatterPlotConfig()

    # Extract data
    x_energies, y_energies = extract_scatter_data(
        structures,
        energy_source1,
        energy_source2,
        data_type=config.data_type,
        x_range=config.x_range,
        y_range=config.y_range,
    )

    # Validate
    if x_energies.size == 0 or y_energies.size == 0:
        raise ValueError(
            f"No valid energy pairs found for plotting "
            f"({energy_source1} vs {energy_source2})"
        )

    # Render plot
    fig, ax = _render_energy_scatter_plot(
        x_energies,
        y_energies,
        energy_source1,
        energy_source2,
        config,
        label=label,
        color=color,
    )

    # Save if requested
    if save_path:
        save_plot_to_file(fig, save_path, dpi=config.dpi)

    return fig


def plot_energy_vs_scatter(
    structures: List[Dict],
    energy_source1: str,
    energy_source2: str,
    data_type: str = "value",
    dpi: int | None = None,
    save_path: Path | None = None,
    figsize: tuple = (8, 6),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    ax: Axes | None = None,
    fig: Figure | None = None,
    label: str | None = None,
    color: str | None = None,
    show_diagonal: bool = True,
) -> Tuple[Figure, Axes, np.ndarray, np.ndarray]:
    """
    Plot formation energy scatter comparing two energy sources (legacy interface).

    This function supports building multi-series plots by reusing axes.
    Returns data arrays for stats calculation.

    Parameters
    ----------
    structures : List[Dict]
        List of structure result dictionaries
    energy_source1 : str
        First energy source identifier (X-axis)
    energy_source2 : str
        Second energy source identifier (Y-axis)
    data_type : str, optional
        Data type to extract ('value' or 'delta'), by default "value"
    dpi : int | None, optional
        Figure DPI, by default None
    save_path : Path | None, optional
        Path to save figure, by default None
    figsize : tuple, optional
        Figure size, by default (8, 6)
    xlim : tuple[float, float] | None, optional
        X-axis limits, by default None
    ylim : tuple[float, float] | None, optional
        Y-axis limits, by default None
    x_range : tuple[float, float] | None, optional
        Physical range filter for X-axis (min, max), by default None
    y_range : tuple[float, float] | None, optional
        Physical range filter for Y-axis (min, max), by default None
    ax : Axes | None, optional
        Existing axes to plot on, by default None
    fig : Figure | None, optional
        Existing figure, by default None
    label : str | None, optional
        Label for legend, by default None
    color : str | None, optional
        Scatter point color, by default None (auto-cycle)
    show_diagonal : bool, optional
        Show diagonal reference line, by default True

    Returns
    -------
    Tuple[Figure, Axes, np.ndarray, np.ndarray]
        Figure, axes, x_energies, y_energies

    Raises
    ------
    ValueError
        If no valid energy pairs are found
    """
    # Extract data
    x_energies, y_energies = extract_scatter_data(
        structures, energy_source1, energy_source2, data_type, x_range, y_range
    )

    # Validate
    if x_energies.size == 0 or y_energies.size == 0:
        raise ValueError(
            f"No valid energy pairs found for plotting "
            f"({energy_source1} vs {energy_source2})"
        )

    # Build config from parameters
    config = ScatterPlotConfig(
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
        dpi=dpi,
        show_diagonal=show_diagonal,
    )

    # Render plot
    fig, ax = _render_energy_scatter_plot(
        x_energies,
        y_energies,
        energy_source1,
        energy_source2,
        config,
        ax=ax,
        fig=fig,
        label=label,
        color=color,
    )

    # Save if requested
    if save_path:
        save_plot_to_file(fig, save_path, dpi=dpi)

    return fig, ax, x_energies, y_energies
