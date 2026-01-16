import logging

from vsf.calculators.base import BaseNN

# These imports trigger calculator registration via decorators
from vsf.calculators.custom import *
from vsf.calculators.registry import get_calculator_registry
from vsf.energy.energy_source import EnergySource

LOGGER = logging.getLogger(__name__)

# Optional instance cache for performance
_CALCULATOR_INSTANCE_CACHE = {}


def get_calculator(
    source: EnergySource,
    device: str = "cpu",
    auto_init: bool = True,
    use_cache: bool = True,
) -> BaseNN:
    """Get a calculator instance for the specified energy source."""

    # Check cache first if enabled
    cache_key = (source, device)
    if use_cache and cache_key in _CALCULATOR_INSTANCE_CACHE:
        calculator = _CALCULATOR_INSTANCE_CACHE[cache_key]
        LOGGER.debug(f"Using cached calculator for {source.name}")
        return calculator

    # Get the appropriate calculator class
    registry = get_calculator_registry()
    if source not in registry:
        raise ValueError(f"No calculator available for EnergySource: {source}")

    calculator_class = registry[source]
    calculator = calculator_class(device=device)

    # Set the energy source
    calculator.energy_source = source

    # Initialize if requested
    if auto_init:
        calculator.initialize()

    # Cache the instance if caching is enabled
    if use_cache:
        _CALCULATOR_INSTANCE_CACHE[cache_key] = calculator

    return calculator


def clear_calculator_cache() -> None:
    """Clear the calculator instance cache."""
    _CALCULATOR_INSTANCE_CACHE.clear()
