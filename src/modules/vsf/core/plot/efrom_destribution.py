import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .utils import set_plot_defaults

LOGGER = logging.getLogger(__name__)


@dataclass
class HeatmapPlotConfig:
    """Configuration for binary alloy heatmap plots."""

    figsize: tuple[int, int] = (10, 8)
    vmin: float | None = None
    vmax: float | None = None
    cmap: str | None = None
    annot: bool = False
    dpi: int | None = None
    show_triangle: str = "lower"
    bar_title: str = "Formation Energy (eV/atom)"


def extract_binary_heatmap_data(
    structures: List[Dict],
    energy_source: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract and process binary alloy formation energy data.

    Parameters
    ----------
    structures : List[Dict]
        List of structure result dictionaries
    energy_source : str
        Energy source identifier

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Matrix of formation energies and list of elements

    Notes
    -----
    Always extracts the 'value' field from results. If delta (relative to reference)
    is needed, preprocess structures upstream using prepare_structures_for_delta_plotting().
    """
    binary_data = defaultdict(list)  # {(elem1, elem2): [energies]}

    for structure in structures:
        # Extract formation energy
        formation_data = structure.get("formation_energy", {})
        results = formation_data.get("results", {})
        if energy_source not in results:
            continue

        formation_energy = results[energy_source]["value"]

        # Extract composition - try multiple common formats
        elements = _extract_elements_from_structure(structure)

        # Filter to binary systems only
        if len(elements) != 2:
            continue

        # Create canonical pair (alphabetically ordered)
        pair = tuple(sorted(elements))
        binary_data[pair].append(formation_energy)

        LOGGER.debug(f"Found binary {pair}: {formation_energy:.6f} eV/atom")

    if not binary_data:
        LOGGER.warning("No binary alloy structures found")
        return np.array([]), []

    # Calculate averages for each pair
    averaged_data = {}
    for pair, energies in binary_data.items():
        avg_energy = sum(energies) / len(energies)
        averaged_data[pair] = avg_energy
        LOGGER.debug(
            f"Binary pair {pair}: {len(energies)} structures, avg = {avg_energy:.6f} eV/atom"
        )

    LOGGER.info(
        f"Processed {len(binary_data)} binary alloy pairs from {len(structures)} structures"
    )

    # Build matrix
    all_elements = sorted(set(elem for pair in averaged_data.keys() for elem in pair))
    n_elements = len(all_elements)
    matrix = np.full((n_elements, n_elements), np.nan)

    element_to_index = {elem: idx for idx, elem in enumerate(all_elements)}

    for (elem1, elem2), avg_energy in averaged_data.items():
        i, j = element_to_index[elem1], element_to_index[elem2]
        matrix[i, j] = avg_energy
        matrix[j, i] = avg_energy  # Symmetric matrix

    return np.abs(matrix), all_elements


def _extract_elements_from_structure(structure: Dict) -> List[str]:
    """
    Extract unique elements from structure data.

    Parameters
    ----------
    structure : Dict
        Structure result dictionary

    Returns
    -------
    List[str]
        List of unique element symbols
    """
    # Try common structure formats

    # Method 1: Check for composition field
    if "composition" in structure:
        comp = structure["composition"]
        if isinstance(comp, dict):
            return list(comp.keys())

    # Method 2: Parse formula field
    if "formula" in structure:
        return _parse_formula_for_elements(structure["formula"])

    # Method 3: Check for elements field
    if "elements" in structure:
        elements = structure["elements"]
        if isinstance(elements, list):
            return list(set(elements))

    # Method 4: Try to extract from name/path
    if "name" in structure:
        return _parse_name_for_elements(structure["name"])

    LOGGER.warning(
        f"Could not extract elements from structure: {structure.get('name', 'unnamed')}"
    )
    return []


def _parse_formula_for_elements(formula: str) -> List[str]:
    """Parse chemical formula to extract unique elements."""
    import re

    # Simple regex to find element symbols (capital letter followed by optional lowercase)
    elements = re.findall(r"[A-Z][a-z]?", formula)
    return list(set(elements))


def _parse_name_for_elements(name: str) -> List[str]:
    """Try to extract elements from structure name/path."""
    # Look for common naming patterns like "AgAu_structure" or "Ni50Cu50"
    import re

    elements = re.findall(r"[A-Z][a-z]?", name)
    # Filter out common non-element patterns
    element_symbols = [elem for elem in elements if len(elem) <= 2 and elem.isalpha()]
    return list(set(element_symbols))


def _prepare_structures_for_delta_plotting(
    structures: List[Dict],
    reference_source: str,
    property_name: str = "formation_energy",
) -> List[Dict]:
    """
    Preprocess structures by computing deltas relative to a reference source.

    Replaces the 'value' field in results with (gnn_value - reference_value).
    The plotter then extracts these pre-computed deltas as if they were raw values.

    Parameters
    ----------
    structures : List[Dict]
        List of structure result dictionaries
    reference_source : str
        Source key to use as reference (e.g., "VASP")
    property_name : str
        Which analyzer to process ("formation_energy", "potential_energy", etc.)

    Returns
    -------
    List[Dict]
        Structures with value fields replaced by computed deltas

    Raises
    ------
    ValueError
        If reference_source not found in any structure's results
    """
    for structure in structures:
        results = structure[property_name]["results"]

        if reference_source not in results:
            raise ValueError(
                f"Reference source '{reference_source}' not found in {structure['name']}"
            )

        ref_value = results[reference_source]["value"]

        for source, result in results.items():
            if source == reference_source:
                result["value"] = 0.0  # Delta of reference to itself is 0
            else:
                result["value"] = result["value"] - ref_value

    return structures


def _render_binary_heatmap_plot(
    matrix: np.ndarray,
    elements: List[str],
    title: str,
    config: HeatmapPlotConfig,
) -> Figure:
    """
    Create binary alloy heatmap visualization.

    Parameters
    ----------
    matrix : np.ndarray
        Formation energy matrix
    elements : List[str]
        Element labels
    title : str
        Plot title
    config : HeatmapPlotConfig
        Configuration for heatmap appearance and layout

    Returns
    -------
    Figure
        The matplotlib figure

    Raises
    ------
    ValueError
        If show_triangle has invalid value
    """
    set_plot_defaults()

    fig, ax = plt.subplots(figsize=config.figsize)

    # Create mask for triangle selection
    mask = None
    if config.show_triangle == "upper":
        # Mask the lower triangle (excluding diagonal)
        mask = np.tril(np.ones_like(matrix, dtype=bool), k=-1)
    elif config.show_triangle == "lower":
        # Mask the upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    elif config.show_triangle != "full":
        raise ValueError(
            f"show_triangle must be 'upper', 'lower', or 'full', got '{config.show_triangle}'"
        )

    cmap = config.cmap
    if cmap is None:
        full_cmap = matplotlib.colormaps["magma_r"]
        cmap = ListedColormap(full_cmap(np.linspace(0, 0.65, 256)))

    # Create heatmap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("left", size="3%", pad=0.85)

    sns.heatmap(
        matrix,
        xticklabels=elements,
        yticklabels=elements,
        cmap=cmap,
        annot=config.annot,
        fmt=".3f" if config.annot else "",
        vmin=config.vmin,
        vmax=config.vmax,
        mask=mask,
        cbar_ax=cax,
        cbar_kws={"label": config.bar_title},
        ax=ax,
    )

    # flip ticks to the left side
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")

    ax.set_title(title)
    ax.set_xlabel("Element")
    ax.set_ylabel("Element")
    ax.set_aspect("equal")

    # Rotate labels for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    return fig


def plot_formation_energy_heatmap(
    structures: List[Dict],
    energy_source: str,
    config: HeatmapPlotConfig | None = None,
) -> Figure:
    """
    Plot formation energy heatmap for binary alloys.

    Parameters
    ----------
    structures : List[Dict]
        List of structure result dictionaries
    energy_source : str
        Energy source identifier (e.g., "VASP", "MACE")
    config : HeatmapPlotConfig, optional
        Plot configuration (uses defaults if None)

    Returns
    -------
    Figure
        Matplotlib figure object

    Raises
    ------
    ValueError
        If no binary alloy data found or rendering fails
    """
    if config is None:
        config = HeatmapPlotConfig()

    matrix, elements = extract_binary_heatmap_data(structures, energy_source)

    if matrix.size == 0:
        raise ValueError("No binary alloy data found for heatmap")

    return _render_binary_heatmap_plot(
        matrix,
        elements,
        f"Formation Energy Heatmap - {energy_source}",
        config,
    )


def plot_delta_heatmap(
    structures: List[Dict],
    energy_source: str,
    reference_source: str = "VASP",
    config: HeatmapPlotConfig | None = None,
) -> Figure:
    """
    Plot delta (error) heatmap for binary alloys.

    Shows differences between energy_source and reference_source predictions.

    Parameters
    ----------
    structures : List[Dict]
        List of structure result dictionaries
    energy_source : str
        Energy source to compare (e.g., "MACE", "SevenNet")
    reference_source : str
        Reference energy source (default: "VASP")
    config : HeatmapPlotConfig, optional
        Plot configuration (uses delta defaults if None)

    Returns
    -------
    Figure
        Matplotlib figure object

    Raises
    ------
    ValueError
        If reference_source not found or no binary alloy data found
    """
    if config is None:
        config = HeatmapPlotConfig(
            bar_title="Formation Energy Delta (eV/atom)",
        )

    preprocessed = _prepare_structures_for_delta_plotting(
        structures, reference_source, property_name="formation_energy"
    )
    matrix, elements = extract_binary_heatmap_data(preprocessed, energy_source)

    if matrix.size == 0:
        raise ValueError("No binary alloy data found for delta heatmap")

    return _render_binary_heatmap_plot(
        matrix,
        elements,
        f"Formation Energy Delta - {energy_source} vs {reference_source}",
        config,
    )
