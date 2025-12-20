import json
import logging
from pathlib import Path
from typing import Dict, List

LOGGER = logging.getLogger(__name__)


class ResultsAggregator:
    """Aggregates formation energy results from multiple directories."""

    def __init__(self, directories: List[Path]):
        """
        Initialize aggregator with list of directories to scan.

        Parameters
        ----------
        directories : List[Path]
            List of directories containing structure subdirectories with JSON files
        """
        self.directories = [Path(d) for d in directories]

    def load_all_results(self) -> List[Dict]:
        """
        Scan all directories and load JSON results.

        Returns
        -------
        List[Dict]
            List of structure result dictionaries loaded from JSON files
        """
        all_results = []

        for directory in self.directories:
            LOGGER.info(f"Scanning directory: {directory}")

            if not directory.exists():
                LOGGER.warning(f"Directory does not exist: {directory}")
                continue

            for structure_dir in directory.iterdir():
                if not structure_dir.is_dir():
                    continue

                json_path = structure_dir / f"{structure_dir.name}.json"

                if json_path.exists():
                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        # Add directory info to track origin
                        data["structure_dir"] = str(structure_dir)
                        data["parent_dir"] = str(directory)
                        all_results.append(data)
                    except Exception as e:
                        LOGGER.error(f"Failed to load {json_path}: {e}")

        LOGGER.info(
            f"Loaded {len(all_results)} structure results from {len(self.directories)} directories"
        )
        return all_results

    def get_structure_paths(self) -> Dict[str, Path]:
        """
        Get mapping of structure names to their directory paths.

        Returns
        -------
        Dict[str, Path]
            Mapping of structure names to structure directory paths
        """
        results = self.load_all_results()
        return {result["name"]: Path(result["structure_dir"]) for result in results}


class FormationEnergyFilter:
    """Filters structures by formation energy criteria and outputs paths for bash processing."""

    def filter_by_threshold(
        self, structures: List[Dict], energy_cutoff: float, energy_source: str = "VASP"
    ) -> List[Path]:
        """
        Filter structures by formation energy threshold.

        Parameters
        ----------
        structures : List[Dict]
            List of structure result dictionaries
        energy_cutoff : float
            Formation energy threshold in eV/atom (structures below this are selected)
        energy_source : str
            Energy source identifier (default: 'VASP')

        Returns
        -------
        List[Path]
            List of POSCAR file paths for structures below threshold
        """
        selected_paths = []

        for s in structures:
            # Check if formation energy data exists
            formation_data = s.get("formation_energy", {})
            results = formation_data.get("results", {})
            if energy_source not in results:
                LOGGER.debug(f"No {energy_source} formation energy for {s['name']}")
                continue

            formation_energy = results[energy_source]["value"]

            if formation_energy <= energy_cutoff:
                structure_dir = Path(s["structure_dir"])
                poscar_path = structure_dir / "POSCAR"
                selected_paths.append(poscar_path)
                LOGGER.debug(f"Selected {s['name']}: {formation_energy:.6f} eV/atom")

        LOGGER.info(
            f"Selected {len(selected_paths)} structures below threshold {energy_cutoff} eV/atom"
        )
        return selected_paths

    def filter_by_range(
        self,
        structures: List[Dict],
        min_energy: float,
        max_energy: float,
        energy_source: str = "VASP",
    ) -> List[Path]:
        """
        Filter structures by formation energy range.

        Parameters
        ----------
        structures : List[Dict]
            List of structure result dictionaries
        min_energy : float
            Minimum formation energy in eV/atom (inclusive)
        max_energy : float
            Maximum formation energy in eV/atom (inclusive)
        energy_source : str
            Energy source identifier (default: 'VASP')

        Returns
        -------
        List[Path]
            List of POSCAR file paths for structures in energy range
        """
        selected_paths = []

        for s in structures:
            # Check if formation energy data exists
            formation_data = s.get("formation_energy", {})
            results = formation_data.get("results", {})
            if energy_source not in results:
                LOGGER.debug(f"No {energy_source} formation energy for {s['name']}")
                continue

            formation_energy = results[energy_source]["value"]

            if min_energy <= formation_energy <= max_energy:
                structure_dir = Path(s["structure_dir"])
                poscar_path = structure_dir / "POSCAR"
                selected_paths.append(poscar_path)

        LOGGER.info(
            f"Selected {len(selected_paths)} structures in range [{min_energy}, {max_energy}] eV/atom"
        )
        return selected_paths

    def write_poscar_paths(self, paths: List[Path], output_file: Path) -> None:
        """
        Write POSCAR paths to file for bash consumption.

        Parameters
        ----------
        paths : List[Path]
            List of POSCAR file paths
        output_file : Path
            Output file to write paths to
        """
        output_file = Path(output_file)

        with open(output_file, "w") as f:
            for path in paths:
                f.write(f"{path}\n")

        LOGGER.info(f"Wrote {len(paths)} POSCAR paths to {output_file}")

    def get_formation_energy_stats(
        self, structures: List[Dict], energy_cutoff: float, energy_source: str = "VASP"
    ) -> Dict:
        """
        Get statistical summary of formation energies.

        Parameters
        ----------
        structures : List[Dict]
            List of structure result dictionaries
        energy_cutoff : float
            Energy threshold in eV/atom for counting stable structures
        energy_source : str
            Energy source identifier (default: 'VASP')

        Returns
        -------
        Dict
            Statistical summary including min, max, mean, count, and stability metrics
        """
        energies = []

        for s in structures:
            formation_data = s.get("formation_energy", {})
            results = formation_data.get("results", {})
            if energy_source in results:
                value = results[energy_source]["value"]
                energies.append(value)
            else:
                LOGGER.warning(f"No {energy_source} energy found for {s.get('name')}")

        if not energies:
            return {
                "count": 0,
                "message": f"No {energy_source} formation energies found",
            }

        return {
            "energy_cutoff": energy_cutoff,
            "count": len(energies),
            "min": min(energies),
            "max": max(energies),
            "mean": sum(energies) / len(energies),
            "structures_below_cutoff": sum(1 for e in energies if e <= energy_cutoff),
        }
