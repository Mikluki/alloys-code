import logging
from abc import ABC, abstractmethod
from pathlib import Path

from ..energy.energy_source import EnergySource

LOGGER = logging.getLogger(__name__)


class BaseNN(ABC):
    """Abstract base class for all calculators.

    Designed for two-phase initialization:
    1. Construct the calculator (lightweight, before venv switch)
    2. Call initialize() after switching to the appropriate venv
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
    ):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._ase_calculator = None
        self._energy_source = None

    @property
    def energy_source(self) -> EnergySource:
        """Get the energy source associated with this calculator."""
        if self._energy_source is None:
            raise RuntimeError("Energy source not set for this calculator.")
        return self._energy_source

    @energy_source.setter
    def energy_source(self, value: EnergySource) -> None:
        """Set the energy source for this calculator."""
        self._energy_source = value

    def initialize(self, checkpoint_path: str | Path | None = None, **kwargs) -> None:
        """Initialize the ASE calculator after venv switch.

        Args:
            checkpoint_path: Optional override for the checkpoint path set in __init__
            **kwargs: Additional arguments passed to the subclass initializer
        """
        if self._ase_calculator is not None:
            raise RuntimeError("Calculator already initialized")

        # Allow override at initialization time
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)

        LOGGER.info(f"Loading [{self.__class__.__name__}] on {self.device}")
        if kwargs:
            LOGGER.info(f"Kwargs used: {kwargs}")

        self._initialize_custom_subclass(**kwargs)

    @abstractmethod
    def _initialize_custom_subclass(self, **kwargs) -> None:
        """Subclass-specific implementation for initialization."""
        pass

    @property
    def ase_calculator(self):
        """Get the underlying ASE calculator instance."""
        if self._ase_calculator is None:
            raise RuntimeError("Calculator not initialized. Call initialize() first.")
        return self._ase_calculator

    def get_model_info(self) -> dict:
        """Return metadata about the model being used."""
        return {
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "device": self.device,
            "calculator_type": self.__class__.__name__,
        }
