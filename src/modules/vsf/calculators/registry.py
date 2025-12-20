import logging
from typing import Dict, Type

from ..energy.energy_source import EnergySource
from .base import BaseNN

LOGGER = logging.getLogger(__name__)
# Private registry cache
_CALCULATOR_REGISTRY: Dict[EnergySource, Type[BaseNN]] = {}


def register_calculator(
    energy_source: EnergySource, calculator_class: Type[BaseNN]
) -> None:
    """Register a calculator class for a specific energy source."""

    if energy_source in _CALCULATOR_REGISTRY:
        LOGGER.warning(
            f"Overriding existing calculator for {energy_source.name}: "
            f"{_CALCULATOR_REGISTRY[energy_source].__name__} → {calculator_class.__name__}"
        )

    _CALCULATOR_REGISTRY[energy_source] = calculator_class


def get_calculator_registry() -> Dict[EnergySource, Type[BaseNN]]:
    """Get the current mapping of energy sources to calculator classes."""
    # Return a copy to prevent modification
    return _CALCULATOR_REGISTRY.copy()
