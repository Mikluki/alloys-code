"""
Extract individual atomic configurations from VASP XDATCAR files WITH ENERGIES.
Organizes configurations by element for autocorrelation analysis and GNN benchmarking.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ase.io import read as ase_read
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

LOGGER = logging.getLogger(__name__)


@dataclass
class ConfigurationData:
    """Container for extracted configuration with metadata."""

    structure: Structure
    source_dir: Path
    config_index: int  # Configuration number within XDATCAR (1-based)
    time_step: int  # MD time step (config_index * NBLOCK)
    element: str
    energy_sigma_to_0: Optional[float] = None  # E0 - for formation energy calculations
    free_energy: Optional[float] = None  # F - for finite-T thermodynamics
    temperature: Optional[float] = None  # Temperature at this MD step

    @property
    def config_id(self) -> str:
        """Unique identifier for this configuration."""
        return f"{self.source_dir.name}_config_{self.config_index:03d}"


def parse_oszicar(oszicar_path: Path) -> Dict[int, Dict[str, float]]:
    """
    Extract energies from OSZICAR MD run.
    """
    oszicar_path = Path(oszicar_path)

    if not oszicar_path.exists():
        raise FileNotFoundError(f"OSZICAR not found: {oszicar_path}")

    LOGGER.info(f"Parsing OSZICAR: {oszicar_path}")

    energies = {}

    ### >> Use \d*\. instead of \d+\. to handle .25 format (no leading zero)
    pattern = re.compile(
        r"^\s*(\d+)\s+"  # Step number
        r"T=\s+([\d.]+)\s+"  # Temperature
        r"E=\s+([-+]?\d*\.\d+E[+-]?\d+)\s+"  # E (note \d* not \d+)
        r"F=\s+([-+]?\d*\.\d+E[+-]?\d+)\s+"  # F (note \d* not \d+)
        r"E0=\s+([-+]?\d*\.\d+E[+-]?\d+)\s+"  # E0 (note \d* not \d+)
        r"EK=\s+([-+]?\d*\.\d+E[+-]?\d+)"  # EK (note \d* not \d+)
    )

    with open(oszicar_path, "r") as f:
        for line in f:
            if "T=" in line and "E0=" in line:
                match = pattern.match(line)
                if match:
                    step = int(match.group(1))
                    energies[step] = {
                        "T": float(match.group(2)),
                        "E": float(match.group(3)),
                        "F": float(match.group(4)),
                        "E0": float(match.group(5)),
                        "EK": float(match.group(6)),
                    }

    if not energies:
        LOGGER.error(f"No MD energy lines found in {oszicar_path.name}")
    else:
        LOGGER.info(f"  Successfully extracted energies for {len(energies)} MD steps")

    return energies


def parse_xdatcar(xdatcar_path: Path) -> List[ConfigurationData]:
    """
    Parse XDATCAR file and extract all configurations.

    Args:
        xdatcar_path: Path to XDATCAR file

    Returns:
        List of ConfigurationData objects (without energies)
    """
    xdatcar_path = Path(xdatcar_path)

    if not xdatcar_path.exists():
        raise FileNotFoundError(f"XDATCAR not found: {xdatcar_path}")

    LOGGER.info(f"Parsing XDATCAR: {xdatcar_path}")

    with open(xdatcar_path, "r") as f:
        lines = f.readlines()

    # Parse header
    system_name = lines[0].strip()
    scale_factor = float(lines[1].strip())

    # Lattice vectors
    lattice_matrix = []
    for i in range(2, 5):
        lattice_matrix.append([float(x) for x in lines[i].split()])
    lattice_matrix = np.array(lattice_matrix) * scale_factor
    lattice = Lattice(lattice_matrix)

    # Element info
    elements = lines[5].split()
    element_counts = [int(x) for x in lines[6].split()]

    # Build species list
    species = []
    for elem, count in zip(elements, element_counts):
        species.extend([elem] * count)

    # Detect primary element
    element_dict = dict(zip(elements, element_counts))
    primary_element = max(element_dict, key=element_dict.get)  # type: ignore

    total_atoms = sum(element_counts)

    LOGGER.info(f"  System: {system_name}")
    LOGGER.info(f"  Elements: {dict(zip(elements, element_counts))}")
    LOGGER.info(f"  Primary element: {primary_element}")
    LOGGER.info(f"  Total atoms: {total_atoms}")

    # Extract configurations
    configurations = []
    config_index = 1
    i = 7

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Direct configuration="):
            config_num = int(line.split("=")[1].strip())

            coords = []
            for j in range(i + 1, i + 1 + total_atoms):
                if j >= len(lines):
                    LOGGER.error(
                        f"Incomplete configuration {config_num}, stopping extraction"
                    )
                    break
                coord_line = lines[j].strip()
                if coord_line:
                    coords.append([float(x) for x in coord_line.split()[:3]])

            if len(coords) == total_atoms:
                structure = Structure(lattice, species, coords)
                time_step = config_index * 100  # Assuming NBLOCK=100

                config_data = ConfigurationData(
                    structure=structure,
                    source_dir=xdatcar_path.parent,
                    config_index=config_index,
                    time_step=time_step,
                    element=primary_element,
                )

                configurations.append(config_data)
                config_index += 1
            else:
                LOGGER.error(f"Skipping incomplete configuration {config_num}")

            i += total_atoms + 1
        else:
            i += 1

    LOGGER.info(f"  Extracted {len(configurations)} configurations")
    return configurations


def parse_xdatcar_with_energies(
    xdatcar_path: Path, oszicar_path: Path
) -> List[ConfigurationData]:
    """
    Parse XDATCAR and attach energies from OSZICAR.

    Args:
        xdatcar_path: Path to XDATCAR file
        oszicar_path: Path to OSZICAR file

    Returns:
        List of ConfigurationData objects with energies attached
    """
    # Get energies from OSZICAR
    try:
        energies = parse_oszicar(oszicar_path)
    except Exception as e:
        LOGGER.error(f"Failed to parse OSZICAR at {oszicar_path}: {e}")
        energies = {}

    # Parse XDATCAR configurations
    configurations = parse_xdatcar(xdatcar_path)

    # Attach energies
    matched = 0
    for config in configurations:
        step = config.config_index
        if step in energies:
            config.energy_sigma_to_0 = energies[step]["E0"]
            config.free_energy = energies[step]["F"]
            config.temperature = energies[step]["T"]
            matched += 1
        else:
            LOGGER.error(f"No energy found for {config.source_dir.name} config {step}")

    LOGGER.info(
        f"  Matched energies for {matched}/{len(configurations)} configurations "
        f"({matched/len(configurations)*100:.1f}%)"
    )

    return configurations


def extract_all_frames_from_vasprun(
    base_dir: Path,
) -> Dict[str, List[ConfigurationData]]:
    """
    Extract ALL frames from vasprun.xml files (not XDATCAR).

    This reads every MD step, not just the NBLOCK-subsampled frames.
    Essential for proper burn-in detection and decorrelation analysis.

    Args:
        base_dir: Directory containing MD run directories with vasprun.xml

    Returns:
        Dictionary mapping element -> list of configurations
    """
    LOGGER.info(f"Searching for vasprun.xml files in {base_dir}")

    # Find all vasprun.xml files
    vasprun_paths = list(base_dir.rglob("**/vasprun.xml"))
    LOGGER.info(f"Found {len(vasprun_paths)} vasprun.xml files")

    all_configs = []
    failed_extractions = []

    adaptor = AseAtomsAdaptor()

    for vasprun_path in vasprun_paths:
        source_dir = vasprun_path.parent

        try:
            # Read all frames using ASE
            LOGGER.info(f"Reading {source_dir.name}...")
            atoms_list = ase_read(str(vasprun_path), index=":")

            # Handle single frame case
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]

            LOGGER.info(f"  Found {len(atoms_list)} frames")

            # Get element from first frame
            symbols = atoms_list[0].get_chemical_symbols()
            unique_symbols = set(symbols)

            if len(unique_symbols) != 1:
                LOGGER.warning(
                    f"  Multiple elements in {source_dir.name}: {unique_symbols}. Skipping."
                )
                continue

            element = list(unique_symbols)[0]

            # Convert each frame to ConfigurationData
            for config_idx, atoms in enumerate(atoms_list, start=1):
                # Convert ASE Atoms to pymatgen Structure
                structure = adaptor.get_structure(atoms)

                # Extract energy if available
                energy = None
                try:
                    energy = atoms.get_potential_energy(apply_constraint=False)
                except Exception:
                    pass

                # Create ConfigurationData
                # Time step is approximated from index (exact value in vasprun but harder to extract)
                config = ConfigurationData(
                    structure=structure,
                    source_dir=source_dir,
                    config_index=config_idx,
                    time_step=config_idx,  # Approximate; actual timestep would need XML parsing
                    element=element,
                    energy_sigma_to_0=energy,
                    free_energy=None,  # Could extract from vasprun if needed
                    temperature=None,
                )

                all_configs.append(config)

            LOGGER.info(f"  Extracted {len(atoms_list)} frames for {element}")

        except Exception as e:
            LOGGER.error(f"Failed to extract {source_dir.name}: {e}")
            failed_extractions.append(source_dir)
            continue

    # Group by element
    configs_by_element = {}
    for config in all_configs:
        element = config.element
        if element not in configs_by_element:
            configs_by_element[element] = []
        configs_by_element[element].append(config)

    # Summary
    LOGGER.info("=" * 60)
    LOGGER.info("EXTRACTION SUMMARY")
    LOGGER.info("=" * 60)
    for element, configs in configs_by_element.items():
        with_energy = sum(1 for c in configs if c.energy_sigma_to_0 is not None)
        energy_pct = (with_energy / len(configs) * 100) if configs else 0
        LOGGER.info(
            f"{element}: {len(configs)} configs, "
            f"{with_energy} with energies ({energy_pct:.1f}%)"
        )

    total_configs = sum(len(configs) for configs in configs_by_element.values())
    LOGGER.info(f"Total configurations: {total_configs}")

    if failed_extractions:
        LOGGER.error(f"Failed extractions: {len(failed_extractions)}")

    return configs_by_element
