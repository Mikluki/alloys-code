"""
Filter and transform configuration data.
Pure functions with no I/O side effects.
"""

import logging
from typing import List, Tuple

from vsf.liquid.diagnostic.autocorrelation import AutocorrelationResult
from vsf.liquid.extract import ConfigurationData

LOGGER = logging.getLogger(__name__)


def filter_decorrelated(
    configs: List[ConfigurationData], autocorr_result: AutocorrelationResult
) -> List[ConfigurationData]:
    """
    Filter configurations to only decorrelated ones.

    Args:
        configs: List of all configurations for an element
        autocorr_result: Autocorrelation analysis result for that element

    Returns:
        List of only decorrelated configurations
    """
    sorted_configs = sorted(configs, key=lambda x: x.time_step)
    decorrelated = []

    for idx in autocorr_result.decorrelated_indices:
        if idx < len(sorted_configs):
            decorrelated.append(sorted_configs[idx])
        else:
            LOGGER.warning(
                f"Decorrelation index {idx} out of range (max: {len(sorted_configs)-1})"
            )

    LOGGER.debug(f"Filtered {len(decorrelated)}/{len(configs)} decorrelated configs")
    return decorrelated


def mark_decorrelated(
    configs: List[ConfigurationData], autocorr_result: AutocorrelationResult
) -> List[Tuple[ConfigurationData, bool]]:
    """
    Mark each configuration as decorrelated or not.

    Args:
        configs: List of configurations
        autocorr_result: Autocorrelation analysis result

    Returns:
        List of (config, is_decorrelated) tuples
    """
    sorted_configs = sorted(configs, key=lambda x: x.time_step)
    decorr_set = set(autocorr_result.decorrelated_indices)

    marked = [(config, i in decorr_set) for i, config in enumerate(sorted_configs)]

    n_decorr = sum(1 for _, is_decorr in marked if is_decorr)
    LOGGER.debug(f"Marked {n_decorr}/{len(configs)} configs as decorrelated")

    return marked


def sort_by_time(configs: List[ConfigurationData]) -> List[ConfigurationData]:
    """Sort configurations by time step."""
    return sorted(configs, key=lambda x: x.time_step)


def sort_by_energy(configs: List[ConfigurationData]) -> List[ConfigurationData]:
    """Sort configurations by energy (E0)."""
    # Filter out configs without energy, then sort
    with_energy = [c for c in configs if c.energy_sigma_to_0 is not None]
    return sorted(with_energy, key=lambda x: x.energy_sigma_to_0)  # type: ignore
