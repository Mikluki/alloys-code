"""
Generate traceability and metadata files.
"""

import logging
from pathlib import Path
from typing import Dict

from vsf.liquid.diagnostic.autocorrelation import AutocorrelationResult
from vsf.liquid.extract import ConfigurationData

LOGGER = logging.getLogger(__name__)


def generate_traceability_file(
    organized_mapping: Dict[str, Path],
    id_to_config: Dict[str, ConfigurationData],
    autocorr_results: Dict[str, AutocorrelationResult],
    output_path: Path,
):
    """
    Generate comprehensive traceability file.

    Args:
        organized_mapping: Structure_ID -> directory path mapping
        id_to_config: Structure_ID -> ConfigurationData mapping
        autocorr_results: Element -> AutocorrelationResult mapping
        output_path: Where to save traceability file
    """
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write("STRUCTURE TRACEABILITY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Maps structure IDs back to original VASP MD simulations\n")
        f.write("Each structure has: POSCAR + energy.json with full metadata\n\n")

        # Summary by element
        f.write("SUMMARY BY ELEMENT:\n")
        f.write("-" * 80 + "\n")
        for element, result in sorted(autocorr_results.items()):
            f.write(f"{element}:\n")
            f.write(f"  Decorrelation time (τ): {result.decorrelation_time} steps\n")
            f.write(f"  Total configs: {result.total_configs}\n")
            f.write(f"  Decorrelated configs: {result.n_decorrelated}\n")
            f.write(f"  Sampling efficiency: {result.sampling_efficiency:.1%}\n\n")

        # Detailed mapping
        f.write("\nDETAILED MAPPING:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Structure_ID':<18} {'Element':<8} {'Source_Dir':<25} "
            f"{'Config':<8} {'Time_Step':<10} {'Energy_E0':<12}\n"
        )
        f.write("-" * 80 + "\n")

        for struct_id in sorted(organized_mapping.keys()):
            if struct_id in id_to_config:
                config = id_to_config[struct_id]
                energy_str = (
                    f"{config.energy_sigma_to_0:.4f}"
                    if config.energy_sigma_to_0 is not None
                    else "N/A"
                )
                f.write(
                    f"{struct_id:<18} {config.element:<8} "
                    f"{config.source_dir.name:<25} {config.config_index:<8} "
                    f"{config.time_step:<10} {energy_str:<12}\n"
                )

        f.write("\n")
        f.write("USAGE:\n")
        f.write("- Each structure_id maps to a directory: target_dir/structure_id/\n")
        f.write("- Each directory contains: POSCAR + energy.json\n")
        f.write("- energy.json has full traceability and energy data\n")
        f.write("- Use structure_id to reference specific configurations\n")

    LOGGER.info(f"Traceability file created: {output_path}")


def generate_quick_lookup(
    id_to_config: Dict[str, ConfigurationData], output_path: Path
):
    """
    Generate quick lookup file for easy reference.

    Args:
        id_to_config: Structure_ID -> ConfigurationData mapping
        output_path: Where to save lookup file
    """
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write("# Quick Lookup: Structure_ID -> Source\n")
        f.write("# Format: structure_id source_dir/config_index\n\n")

        for struct_id in sorted(id_to_config.keys()):
            config = id_to_config[struct_id]
            f.write(
                f"{struct_id} {config.source_dir.name}/config_{config.config_index:03d}\n"
            )

    LOGGER.info(f"Quick lookup file created: {output_path}")


def generate_summary_stats(
    autocorr_results: Dict[str, AutocorrelationResult], output_path: Path
):
    """
    Generate statistical summary of decorrelation analysis.

    Args:
        autocorr_results: Element -> AutocorrelationResult mapping
        output_path: Where to save summary file
    """
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        f.write("DECORRELATED CONFIGURATION SAMPLING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            "Analysis method: Position autocorrelation using statsmodels.tsa.acf()\n"
        )
        f.write("Decorrelation criterion: Correlation ≤ 1/e ≈ 0.37\n")
        f.write("Sampling strategy: Every τ time steps\n\n")

        total_original = sum(
            result.total_configs for result in autocorr_results.values()
        )
        total_decorrelated = sum(
            result.n_decorrelated for result in autocorr_results.values()
        )
        overall_efficiency = (
            total_decorrelated / total_original if total_original > 0 else 0
        )

        f.write("RESULTS BY ELEMENT:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Element':<10} {'Original':<12} {'Decorr_τ':<12} "
            f"{'Selected':<12} {'Efficiency':<12}\n"
        )
        f.write("-" * 80 + "\n")

        for element, result in sorted(autocorr_results.items()):
            f.write(
                f"{element:<10} {result.total_configs:<12} {result.decorrelation_time:<12} "
                f"{result.n_decorrelated:<12} {result.sampling_efficiency:<12.1%}\n"
            )

        f.write("-" * 80 + "\n")
        f.write(
            f"{'TOTAL':<10} {total_original:<12} {'-':<12} "
            f"{total_decorrelated:<12} {overall_efficiency:<12.1%}\n"
        )

        f.write(
            f"\nRecommendation: Use {total_decorrelated} decorrelated configurations "
            "for downstream applications.\n"
        )

    LOGGER.info(f"Summary statistics created: {output_path}")
