import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import numpy.typing as npt

from ...energy.energy_source import EnergySource

LOGGER = logging.getLogger(__name__)


@dataclass
class StressResult:
    """Stress tensor result. Pressure is computed on-the-fly."""

    stress_array: npt.NDArray

    @property
    def pressure(self) -> float:
        """Compute hydrostatic pressure from stress tensor.

        P = -1/3 * Tr(σ)
        Positive => compression, negative => tension.

        Returns
        -------
        float
            Hydrostatic pressure in eV/Å³.
        """
        trace = self.stress_array[0] + self.stress_array[1] + self.stress_array[2]
        return -trace / 3.0

    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return {
            "stress_array": self.stress_array.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StressResult":
        """Create result from dictionary for deserialization."""
        return cls(stress_array=np.array(data["stress_array"]))


class StressAnalyzer:
    """Handles stress tensor storage and analysis."""

    def __init__(self) -> None:
        super().__init__()
        self._results: Dict[EnergySource, StressResult] = {}
        self.name = "Stress"

    def add(self, source: EnergySource, stress_array: npt.NDArray) -> None:
        """Add a stress tensor associated with a specific energy source.

        Args:
            source (EnergySource): The energy source for which the stress tensor is added.
            stress_array (npt.NDArray): The 6-component stress tensor in Voigt notation,
                ordered as [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy].
                Units are typically eV/Å³ if obtained from ASE or DFT calculations.
        """
        result = StressResult(stress_array=stress_array)
        self._results[source] = result

    def get(self, source: EnergySource) -> StressResult | None:
        """Get result for a specific source.

        Args:
            source: EnergySource to get results for

        Returns:
            StressResult for the specified source, or None if not found
        """
        return self._results.get(source)

    def get_results(self) -> Dict[EnergySource, StressResult]:
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

    def log_results(self) -> None:
        """Log results for all energy sources."""
        for source, result in self._results.items():
            LOGGER.info(
                f"{self.name} `{source.value}` > Pressure: {result.pressure:.3f} eV/Ų"
            )

    def to_dict(self) -> dict:
        """Convert analyzer data to dictionary."""
        return {
            "results": {
                source.to_dict(): result.to_dict()
                for source, result in self._results.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StressAnalyzer":
        """Create analyzer from dictionary."""
        analyzer = cls()

        # Permissive loading - skip deprecated energy sources
        for source_key, result_data in data["results"].items():
            try:
                energy_source = EnergySource.from_dict(source_key)
                analyzer._results[energy_source] = StressResult.from_dict(result_data)
            except ValueError as e:
                LOGGER.warning(f"Skipping deprecated energy source '{source_key}': {e}")
                continue

        return analyzer
