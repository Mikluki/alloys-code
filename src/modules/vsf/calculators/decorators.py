import logging
from typing import Type, TypeVar

from ..energy.energy_source import EnergySource
from .base import BaseNN
from .registry import register_calculator

LOGGER = logging.getLogger(__name__)

T = TypeVar("T", bound=Type[BaseNN])


def calculator_for(energy_source: EnergySource):
    """Decorator to register a calculator class for a specific energy source."""

    def decorator(cls: T) -> T:
        register_calculator(energy_source, cls)
        return cls

    return decorator
