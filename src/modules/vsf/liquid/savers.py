"""
Pure I/O functions for saving structures and metadata.
No business logic - just writes files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from pymatgen.io.vasp import Poscar

from vsf.liquid.extract import ConfigurationData

LOGGER = logging.getLogger(__name__)


def save_structure_with_energy(
    config: ConfigurationData, output_dir: Path, metadata_extra: Optional[Dict] = None
) -> Path:
    """
    Save single structure as POSCAR + energy.json.

    Args:
        config: Configuration to save
        output_dir: Directory to create and save files in
        metadata_extra: Optional extra metadata to include in energy.json

    Returns:
        Path to created directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save POSCAR
    poscar = Poscar(config.structure)
    poscar.write_file(output_dir / "POSCAR")

    # Build metadata
    metadata = {
        "source_directory": str(config.source_dir.name),
        "config_index": config.config_index,
        "time_step_fs": config.time_step,
        "element": config.element,
        "n_atoms": len(config.structure),
        "volume_A3": float(config.structure.volume),
        "energies_eV": {
            "E0_sigma_to_0": config.energy_sigma_to_0,
            "F_free_energy": config.free_energy,
        },
        "temperature_K": config.temperature,
        "notes": {
            "formation_energy": "Use E0_sigma_to_0 for formation energy calculations",
            "reference": "E0 is energy extrapolated to zero electronic smearing",
            "configuration": "Liquid MD snapshot - NOT relaxed to 0K",
        },
    }

    # Add extra metadata if provided
    if metadata_extra:
        metadata.update(metadata_extra)

    # Save energy.json
    with open(output_dir / "energy.json", "w") as f:
        json.dump(metadata, f, indent=2)

    LOGGER.debug(f"Saved structure to {output_dir}")
    return output_dir


def generate_simple_dirname(config: ConfigurationData, suffix: str = "") -> str:
    """
    Generate simple directory name for a configuration.

    Args:
        config: Configuration data
        suffix: Optional suffix to append

    Returns:
        Directory name string
    """
    base = f"{config.element}_{config.source_dir.name}_config_{config.config_index:03d}"
    return f"{base}_{suffix}" if suffix else base


def save_poscar_only(config: ConfigurationData, poscar_path: Path) -> Path:
    """
    Save only POSCAR file (used for temporary files).

    Args:
        config: Configuration to save
        poscar_path: Path for POSCAR file

    Returns:
        Path to created POSCAR file
    """
    poscar_path = Path(poscar_path)
    poscar_path.parent.mkdir(parents=True, exist_ok=True)

    poscar = Poscar(config.structure)
    poscar.write_file(poscar_path)

    return poscar_path
