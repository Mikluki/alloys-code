import logging
from dataclasses import dataclass
from typing import Dict

from .energy_source import EnergySource

LOGGER = logging.getLogger(__name__)


@dataclass
class EnergyResult:
    """Energy calculation result."""

    value: float

    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return {
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnergyResult":
        """Create result from dictionary for deserialization."""
        # Remove deprecated fields for backward compatibility
        data.pop("delta", None)
        return cls(**data)


class EnergyAnalyzer:
    """Handles energy calculations and storage."""

    def __init__(self) -> None:
        self._results: Dict[EnergySource, EnergyResult] = {}
        self.name = "Energy"

    def add(self, source: EnergySource, value: float) -> None:
        """Add or update result for a specific source."""
        result = EnergyResult(value=value)
        self._results[source] = result

    def get(self, source: EnergySource) -> EnergyResult | None:
        """Get result for a specific source.

        Args:
            source: EnergySource to get results for

        Returns:
            EnergyResult for the specified source, or None if not found
        """
        return self._results.get(source)

    def get_results(self) -> Dict[EnergySource, EnergyResult]:
        """Get all results."""
        return {source: result for source, result in self._results.items()}

    def delete(self, source: EnergySource) -> bool:
        """Delete result for a specific energy source.

        Args:
            source: EnergySource to delete results for

        Returns:
            True if result was found and deleted, False if not found
        """
        if source in self._results:
            del self._results[source]
            LOGGER.info(f"For `{self.name}` source {source} --deleted")
            return True
        return False

    def get_gnn_results(self) -> Dict[EnergySource, EnergyResult]:
        """Get results from all neural network sources."""
        return {
            source: result
            for source, result in self._results.items()
            if source in EnergySource.neural_networks()
        }

    def get_dft_results(self) -> Dict[EnergySource, EnergyResult]:
        """Get results from all DFT-based sources."""
        return {
            source: result
            for source, result in self._results.items()
            if source in EnergySource.vasp_based()
        }

    def log_results(self) -> None:
        """Log results for all energy sources."""
        for source, result in self._results.items():
            LOGGER.info(f"{self.name} `{source.value}` > Value: {result.value:.3f}")

    def to_dict(self) -> dict:
        """Convert analyzer data to dictionary."""
        return {
            "results": {
                source.to_dict(): result.to_dict()
                for source, result in self._results.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnergyAnalyzer":
        """Create analyzer from dictionary."""
        analyzer = cls()

        # Permissive loading - skip deprecated energy sources
        for source_key, result_data in data["results"].items():
            try:
                energy_source = EnergySource.from_dict(source_key)
                analyzer._results[energy_source] = EnergyResult.from_dict(result_data)
            except ValueError:
                continue

        return analyzer


class FormationEnergyAnalyzer(EnergyAnalyzer):
    """Handles formation energy calculations and storage."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Formation Energy"


class PotentialEnergyAnalyzer(EnergyAnalyzer):
    """Handles potential energy calculations and storage."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Potential Energy"
