from pathlib import Path
from typing import NamedTuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class ChillColors(NamedTuple):
    orange: str = "#E69F00"
    sky_blue: str = "#56B4E9"
    green: str = "#009E73"
    yellow: str = "#F0E442"
    dark_blue: str = "#0072B2"
    red_orange: str = "#D55E00"
    pink: str = "#CC79A7"


CHILL_COLORS = ChillColors()


def set_plot_defaults(
    font_size: int = 12,
    title_size: int = 14,
    label_size: int = 11,
    tick_size: int = 9,
    dpi: int = 300,
    line_width: float = 3.0,
    marker_size: int = 5,
    axes_linewidth: float = 1.0,
    tick_major_size: int = 7,
    tick_minor_size: int = 3,
    tick_width: float = 1.0,
    tick_pad: int = 6,
    axes_labelpad: int = 9,
    axes_titlepad: int = 12,
    linewidth_coeff: float = 1.0,
):
    """Set all matplotlib defaults for plots."""
    plt.rcParams.update(
        {
            # Font sizes
            "font.size": font_size,
            "figure.titlesize": title_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            # Legend
            "legend.fontsize": tick_size,
            "grid.linewidth": 0.5 * linewidth_coeff,
            # Lines and markers
            "lines.linewidth": line_width,
            "lines.markersize": marker_size,
            # Tick directions and visibility
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
            # Major ticks
            "xtick.major.pad": tick_pad,
            "ytick.major.pad": tick_pad,
            "xtick.major.size": tick_major_size,
            "ytick.major.size": tick_major_size,
            "xtick.major.width": tick_width * linewidth_coeff,
            "ytick.major.width": tick_width * linewidth_coeff,
            # Minor ticks
            "xtick.minor.size": tick_minor_size,
            "ytick.minor.size": tick_minor_size,
            "xtick.minor.width": tick_width * linewidth_coeff,
            "ytick.minor.width": tick_width * linewidth_coeff,
            # Axes
            "axes.linewidth": axes_linewidth * linewidth_coeff,
            "axes.autolimit_mode": "round_numbers",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "axes.titlepad": axes_titlepad,
            "axes.labelpad": axes_labelpad,
            # Figure
            "figure.dpi": dpi,
            "figure.titleweight": "normal",
            "font.weight": "normal",
        }
    )
    return label_size, tick_size


def save_plot_to_file(
    fig: Figure, save_path: Path, dpi: int | None, dark: bool = False
) -> None:
    """
    Save plot to file.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save
    output_path : Path
        Output file path
    dpi : int
        DPI for saved figure
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if dark is True:
        facecolor = "#071e24"
        edgecolor = "#051519"
    else:
        edgecolor = None
        facecolor = None

    fig.savefig(
        save_path,
        bbox_inches="tight",
        dpi=dpi,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    plt.close(fig)


def get_light_viridis(white_percent=0.0):
    """Create a lighter version of viridis colormap."""
    viridis = plt.cm.viridis  # pyright: ignore
    colors = viridis(np.linspace(0.3, 1, 256))

    # Mix with white to lighten (0.7 = 70% original, 30% white)
    light_colors = colors * (1 - white_percent) + np.array([1, 1, 1, 1]) * white_percent
    idx = int(white_percent * 10)
    light_colors[:, idx] = colors[:, idx]  # Keep original alpha

    return mcolors.ListedColormap(light_colors)


def get_light_plasma(white_percent=0.0):
    """Create a lighter version of plasma colormap."""
    plasma = plt.cm.plasma  # pyright: ignore
    colors = plasma(np.linspace(0, 1, 256))

    # Mix with white to lighten (0.7 = 70% original, 30% white)
    light_colors = colors * (1 - white_percent) + np.array([1, 1, 1, 1]) * white_percent
    light_colors[:, 3] = colors[:, 3]  # Keep original alpha

    return mcolors.ListedColormap(light_colors)
