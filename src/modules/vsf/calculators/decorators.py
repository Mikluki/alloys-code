import logging
from typing import Type, TypeVar

from vsf.calculators.base import BaseNN
from vsf.calculators.registry import register_calculator
from vsf.energy.energy_source import EnergySource

LOGGER = logging.getLogger(__name__)

T = TypeVar("T", bound=Type[BaseNN])


def calculator_for(energy_source: EnergySource):
    """Decorator to register a calculator class for a specific energy source."""

    def decorator(cls: T) -> T:
        register_calculator(energy_source, cls)
        cls._default_energy_source = energy_source
        return cls

    return decorator
