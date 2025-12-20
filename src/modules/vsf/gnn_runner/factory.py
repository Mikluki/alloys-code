import logging

import numpy as np
import numpy.typing as npt
from pymatgen.core import Structure

from ..calculators.base import BaseNN
from ..calculators.registry import get_calculator_registry
from ..energy.energy_source import EnergySource

LOGGER = logging.getLogger(__name__)


def create_calculator(energy_source: EnergySource, **calc_kwargs) -> BaseNN:
    """
    Create and initialize calculator using existing registry.

    Parameters
    ----------
    energy_source : EnergySource
        Which calculator to create
    **calc_kwargs
        Arguments passed to calculator initialization

    Returns
    -------
    BaseNN
        Initialized calculator instance
    """
    registry = get_calculator_registry()
    if energy_source not in registry:
        raise ValueError(f"No calculator registered for {energy_source.value}")

    calculator_class = registry[energy_source]
    calculator = calculator_class(**calc_kwargs)
    calculator.energy_source = energy_source

    # Remove device from initialize kwargs since it's already set in constructor
    init_kwargs = {k: v for k, v in calc_kwargs.items() if k != "device"}
    calculator.initialize(**init_kwargs)

    LOGGER.info(f"Initialized {calculator_class.__name__} for {energy_source.value}")
    return calculator


def calculate_energy_and_stress(
    structure: Structure, calculator: "BaseNN"
) -> tuple[float, npt.NDArray]:
    """
    Compute energy per atom and the full stress tensor (Voigt) using an ASE-compatible calculator.

    Returns
    -------
    (float, ndarray)
        (energy_per_atom_eV, stress_voigt_eV_per_A3[6])
    """
    atoms = structure.to_ase_atoms()
    atoms.calc = calculator.ase_calculator

    # Trigger one calculation; most calculators fill energy+forces+stress together.
    stress = atoms.get_stress(voigt=True)  # shape (6,), eV/Å³
    total_energy = atoms.get_potential_energy()  # eV (likely cached)

    return total_energy / len(atoms), stress
