"""
High-level workflows for organizing and saving decorrelated configurations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from vsf.liquid.diagnostic.decorrelation import ElementDecorrelationResults
from vsf.liquid.organizers import StructureOrganizer

LOGGER = logging.getLogger(__name__)


def save_decorrelated_structures(
    decorr_results: Dict[str, ElementDecorrelationResults],
    output_dir: Path,
    starting_id: int = 1,
    prefix: str = "liquid",
) -> Dict[str, Path]:
    """
    Save decorrelated structures with unique IDs.

    Only configurations identified as decorrelated by decorrelation analysis
    are saved. These represent statistically independent snapshots from
    equilibrated MD trajectories.

    Args:
        configs_by_element: All configurations grouped by element (not used directly)
        decorr_results: Results from decorrelation analysis
        output_dir: Where to save structures (flat directory structure)
        starting_id: Starting ID for numbering
        prefix: Prefix for structure IDs (e.g., "liquid" -> "liquid_Au_0000001")

    Returns:
        Dictionary mapping structure_id -> directory_path

    Example:
        >>> decorr = analyze_all_elements_decorrelation(configs_by_element)
        >>> saved = save_decorrelated_structures(
        ...     configs_by_element,
        ...     decorr,
        ...     Path("output")
        ... )
        >>> print(f"Saved {len(saved)} structures")
    """
    LOGGER.info("=" * 70)
    LOGGER.info("SAVING DECORRELATED STRUCTURES")
    LOGGER.info("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract decorrelated configs from results
    all_saved_paths = {}
    current_id = starting_id
    stats = {}

    for element, result in decorr_results.items():
        # Get all decorrelated configs for this element
        decorr_configs = result.get_all_decorrelated_configs()

        if not decorr_configs:
            LOGGER.error(f"{element}: No decorrelated configs to save")
            stats[element] = {
                "total": result.total_configs_available,
                "decorrelated": 0,
                "n_trajectories": result.n_trajectories,
                "status": "no_configs",
            }
            continue

        # Use element-specific prefix: e.g., "liquid_Au"
        element_prefix = f"{prefix}_{element}"

        # Save structures
        organizer = StructureOrganizer(current_id, element_prefix)
        saved_paths = organizer.organize_configs(decorr_configs, output_dir)

        all_saved_paths.update(saved_paths)
        current_id += len(saved_paths)

        stats[element] = {
            "total": result.total_configs_available,
            "decorrelated": result.total_configs_selected,
            "n_trajectories": result.n_trajectories,
            "status": "success",
        }

        LOGGER.info(
            f"{element}: saved {len(saved_paths)} structures with prefix '{element_prefix}' "
            f"({result.sampling_efficiency:.1%} efficiency)"
        )

    if not all_saved_paths:
        LOGGER.error("No decorrelated configurations to save!")
        return {}

    # Generate README
    _write_readme(
        stats, decorr_results, output_dir, prefix, starting_id, len(all_saved_paths)
    )

    # Summary
    LOGGER.info("=" * 70)
    LOGGER.info("COMPLETE")
    LOGGER.info("=" * 70)
    LOGGER.info(f"Saved structures: {len(all_saved_paths)}")
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(
        f"Structure IDs: {prefix}_{{element}}_{starting_id:07d} to "
        f"..._{starting_id + len(all_saved_paths) - 1:07d}"
    )

    return all_saved_paths


def _write_readme(
    stats: Dict[str, Dict],
    decorr_results: Dict[str, ElementDecorrelationResults],
    output_dir: Path,
    prefix: str,
    starting_id: int,
    total_saved: int,
):
    """
    Write README.md explaining the dataset and traceability.

    Args:
        stats: Statistics per element
        decorr_results: Decorrelation results for additional details
        output_dir: Output directory
        prefix: Structure ID prefix
        starting_id: Starting ID
        total_saved: Total structures saved
    """
    readme_path = output_dir / "README.md"

    with open(readme_path, "w") as f:
        f.write("# Decorrelated Liquid Structures\n\n")

        # Overview
        f.write("## Overview\n\n")
        f.write(f"**Total structures:** {total_saved}\n\n")
        f.write(
            f"**Structure IDs:** `{prefix}_{{element}}_{starting_id:07d}` to "
            f"`..._{starting_id + total_saved - 1:07d}`\n\n"
        )
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # What is this
        f.write("## What is this dataset?\n\n")
        f.write(
            "This dataset contains statistically independent liquid configurations extracted "
            "from equilibrated MD trajectories. Configurations are selected using:\n\n"
        )
        f.write(
            "1. **Burn-in detection** (pymbar): Identifies equilibration point from potential energy\n"
        )
        f.write(
            "2. **Structural decorrelation** (alchemlyb): Subsamples frames based on structural changes\n\n"
        )
        f.write(
            "Only post-burn-in, decorrelated snapshots are included. Each structure represents "
            "an independent sample suitable for machine learning or statistical analysis.\n\n"
        )

        # Statistics
        f.write("## Decorrelation Statistics\n\n")
        for element, stat in stats.items():
            f.write(f"### {element}\n\n")
            if stat["status"] == "success":
                result = decorr_results.get(element)
                f.write(f"- Total configurations: {stat['total']}\n")
                f.write(f"- MD trajectories: {stat['n_trajectories']}\n")
                f.write(f"- Decorrelated configs saved: {stat['decorrelated']}\n")
                efficiency = stat["decorrelated"] / stat["total"] * 100
                f.write(f"- Overall sampling efficiency: {efficiency:.1f}%\n\n")

                if result:
                    f.write("**Per-trajectory details:**\n\n")
                    for traj in result.trajectory_results:
                        f.write(f"- `{traj.source_dir.name}`:\n")
                        f.write(f"  - Total frames: {traj.total_frames}\n")
                        f.write(f"  - Burn-in: {traj.burn_in_frame} frames\n")
                        f.write(
                            f"  - Selected: {traj.n_selected} frames ({traj.sampling_efficiency:.1%})\n"
                        )
                        f.write(
                            f"  - g_energy: {traj.g_energy:.2f}, g_struct: {traj.g_struct:.2f}\n\n"
                        )
            else:
                f.write(f"- Total configurations: {stat['total']}\n")
                f.write("- **Status:** No decorrelated configs available\n\n")

        # Directory structure
        f.write("## Directory Structure\n\n")
        f.write(
            "Each structure is saved in a flat directory with element name in prefix:\n\n"
        )
        f.write("```\n")
        f.write(f"{prefix}_{{element}}_XXXXXXX/\n")
        f.write("├── POSCAR         # Structure file\n")
        f.write("└── energy.json    # Energy data + metadata\n")
        f.write("```\n\n")
        f.write("**Examples:**\n")
        f.write(f"- `{prefix}_Au_0000001/` - First Au structure\n")
        f.write(f"- `{prefix}_Ag_0000042/` - 42nd Ag structure\n\n")

        # Traceability
        f.write("## How to Trace Origins\n\n")
        f.write(
            "Each structure's `energy.json` contains metadata with origin information:\n\n"
        )
        f.write("- `source_dir`: Original MD run directory\n")
        f.write("- `config_index`: Configuration number in XDATCAR (1-based)\n")
        f.write("- `time_step`: MD timestep\n")
        f.write("- `element`: Element type\n")
        f.write("- `structure_id`: Unique ID assigned by organizer\n\n")

        f.write("**To trace a structure back to its origin:**\n\n")
        f.write("1. Open `energy.json` in the structure directory\n")
        f.write("2. Look for the `metadata` section\n")
        f.write(
            "3. Use `source_dir` + `config_index` to locate frame in original XDATCAR\n\n"
        )

        # Analysis details
        f.write("## Analysis Details\n\n")
        f.write("For decorrelation analysis details and diagnostic plots, see:\n\n")
        f.write(
            "- `x-decorrelation_analysis/decorrelation_summary.txt` - Full statistics\n"
        )
        f.write(
            "- `x-decorrelation_analysis/decorr_*.png` - Diagnostic plots per trajectory\n\n"
        )

        f.write("**Understanding the diagnostic plots:**\n\n")
        f.write("Each trajectory has three panels:\n")
        f.write(
            "1. **Left**: Time series showing burn-in (red line) and selected frames (orange)\n"
        )
        f.write("2. **Middle**: Distribution of structural fluctuations post-burn-in\n")
        f.write(
            "3. **Right**: Autocorrelation function showing decorrelation timescale\n\n"
        )

        # Method explanation
        f.write("## Methodology\n\n")
        f.write(
            "**Fraction-moved metric:** At each frame, we compute the fraction of atoms that moved\n"
        )
        f.write(
            "≥ 0.005 Å over a lag of 10 frames (accounting for periodic boundaries). This metric\n"
        )
        f.write(
            "captures structural rearrangements while being robust to periodic artifacts.\n\n"
        )

        f.write(
            "**Statistical inefficiency (g):** Measures correlation in the time series.\n"
        )
        f.write("- g ≈ 1: Uncorrelated (ideal)\n")
        f.write(
            "- g > 1: Correlated (g=5 means 5 frames needed per independent sample)\n\n"
        )

        f.write(
            "**Subsampling:** Uses `alchemlyb.statistical_inefficiency` to select frames\n"
        )
        f.write(
            "that maximize statistical independence while preserving trajectory coverage.\n"
        )

    LOGGER.info(f"README saved: {readme_path}")
