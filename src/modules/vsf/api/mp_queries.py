import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mp_api.client import MPRester
from pymatgen.io.vasp import Poscar

LOGGER = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """
    Configuration settings for Materials Project search operations.

    This dataclass stores configuration parameters for querying the Materials Project
    database and organizing the resulting data.

    Attributes:
        coefficient_threshold (Optional[float]): Maximum threshold for atomic coefficients.
            If provided, structures with coefficients above this value will be filtered out.
            Defaults to None.
        save_poscars (bool): Whether to save poscars files to disk.
            Defaults to False.
        save_dir (str): Directory path where query results and structures will be saved.
            Defaults to "query--results".
    """

    coefficient_threshold: Optional[float] = None
    save_poscars: bool = False
    save_dir: str = "query--results"


class MaterialsProjectQuery:
    """
    Client for querying and retrieving material data from the Materials Project API.

    This class provides methods to search for materials based on element combinations,
    filter results based on atomic coefficients, and manage query results.

    Attributes:
        api_key (str): Materials Project API key for authentication.
        config (SearchConfig): Configuration for search operations.
        fetched_entries_dict (Dict[str, str]): Dictionary mapping material IDs to their
            corresponding POSCAR filenames.
        pure_elements_dict (Dict[str, str]): Dictionary mapping material IDs to element symbols.

    Examples:
        >>> config = SearchConfig(save_poscars=True)
        >>> query = MaterialsProjectQuery(api_key="your_api_key", config=config)
        >>> pure_elements = query.get_hull_refernce()
        >>> query.set_pure_elements(pure_elements)
        >>> query.query_chemsys_for_binary_compositions()
        >>> query.save_results("mpids.json", "metadata.json")
    """

    def __init__(self, api_key: str, config: Optional[SearchConfig] = None):
        """
        Initialize a Materials Project query client.

        Args:
            api_key (str): Materials Project API key for authentication.
            config (Optional[SearchConfig]): Configuration for search operations.
                If None, default configuration will be used.
        """
        self.api_key = api_key
        self.config = config or SearchConfig()
        self.fetched_entries_dict: Dict[str, str] = {}
        self.pure_elements_dict: Dict[str, str] = {}

    def load_pure_elements_from_file(self, json_path: str) -> Dict[str, str]:
        """
        Load pure elements dictionary from JSON file and return it.

        Args:
            json_path (str): Path to JSON file containing pure elements data.

        Returns:
            Dict[str, str]: Dictionary of pure elements.

        Raises:
            FileNotFoundError: If the specified JSON file does not exist.
            ValueError: If the JSON file contains invalid format.
        """
        try:
            with open(json_path, "r") as f:
                elements_dict = json.load(f)
            return elements_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {json_path} not found")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {json_path}")

    def save_pure_elements_to_json(self, json_path: str) -> None:
        """
        Save pure elements dictionary to a JSON file.

        Args:
            json_path (str): Path where the JSON file will be saved.
        """
        save_path = Path(json_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with save_path.open("w") as f:
            json.dump(self.pure_elements_dict, f, indent=4, sort_keys=True)

    def query_chemsys_for_binary_compositions(self) -> Dict[str, str]:
        """
        Query Materials Project API for binary element combinations and populate self.fetched_entries_dict.

        For each binary combination in pure_elements_dict, this method:
        1. Queries the Materials Project database
        2. Filters materials based on coefficient threshold (if specified)
        3. Populates self.fetched_entries_dict with material_id → composition_poscar_name mappings
        4. Optionally saves poscar files to disk

        Returns:
            Dict[str, str]: Dictionary mapping material IDs to composition POSCAR names.
        """
        # Generate all binary element combinations
        element_combinations = combinations(self.pure_elements_dict.values(), 2)

        with MPRester(api_key=self.api_key) as mpr:
            # Process each binary element combination
            for comb in element_combinations:
                chemsys = "-".join(comb)
                LOGGER.info(f"Querying chemical system: {chemsys}")

                try:
                    # Get all materials for this chemical system
                    summary_doc = mpr.materials.summary.search(chemsys=[chemsys])

                    # Process each material found
                    for doc in summary_doc:
                        material_id = doc.material_id  # type: ignore
                        composition = doc.composition  # type: ignore

                        # Check if material passes concentration threshold
                        if self.config.coefficient_threshold is not None:
                            if not all(
                                amount < self.config.coefficient_threshold
                                for amount in composition.values()  # type: ignore
                            ):
                                LOGGER.info(
                                    f"Skipping {material_id}: concentration exceeds threshold {self.config.coefficient_threshold}"
                                )
                                continue

                        # Generate composition POSCAR name
                        if self.config.save_poscars:
                            # Get full structure and use its formula
                            structure = mpr.get_structure_by_material_id(
                                material_id, conventional_unit_cell=True
                            )
                            formula = structure.formula.replace(" ", "_")  # type: ignore
                            composition_poscar_name = f"{formula}_{material_id}.poscar"

                            # Also save structure to disk
                            LOGGER.info(
                                f"Saving poscar {material_id} to {self.config.save_dir}"
                            )
                            self.save_poscar(
                                structure=structure,
                                material_id=material_id,
                                poscar_name=composition_poscar_name,
                            )
                        else:
                            # Just create a name without retrieving full structure
                            formula = "_".join(
                                [
                                    f"{el}{int(amt)}"
                                    for el, amt in composition.items()  # type: ignore
                                ]
                            )
                            composition_poscar_name = f"{formula}_{material_id}.poscar"

                        # Add to results dictionary
                        LOGGER.info(
                            f"Adding {material_id} → {formula}_{material_id} to results"
                        )
                        self.fetched_entries_dict[material_id] = composition_poscar_name

                except Exception as e:
                    LOGGER.info(f"Error processing {chemsys}: {str(e)}")
                    continue  # Continue with next chemical system

        return self.fetched_entries_dict

    def save_poscar(
        self, structure: Any, material_id: str, poscar_name: Optional[str] = None
    ) -> None:
        """
        Save structure to POSCAR file in the specified directory.

        Args:
            structure: A pymatgen Structure object to save.
            material_id (str): Materials Project ID for the structure.
            poscar_name (Optional[str]): Custom filename for the POSCAR file.
                If None, a name will be generated from the formula and material_id.
        """
        # Create directory if it doesn't exist
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Format the file name
        if poscar_name is None:
            formula = structure.formula.replace(" ", "_")
            poscar_name = f"{formula}_{material_id}.poscar"
        poscar_path = save_path / poscar_name

        # Save as POSCAR
        poscar = Poscar(structure)
        poscar.write_file(poscar_path)

    def save_results(
        self,
        output_file: str = "0-results.json",
        metadata_file: str = "0-metadata.json",
    ) -> Tuple[Path, Path]:
        """
        Save results and metadata to JSON files.

        This method saves the fetched materials data and search metadata to separate
        JSON files in the specified directory.

        Args:
            output_file (str): Filename for the main results JSON file.
            metadata_file (str): Filename for the metadata JSON file.

        Returns:
            Tuple[Path, Path]: Paths to the output and metadata files.
        """
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            output_path = save_path / output_file
            metadata_path = save_path / metadata_file

            with output_path.open("w") as json_file:
                json.dump(
                    self.fetched_entries_dict, json_file, indent=4, sort_keys=True
                )

            metadata = {
                "struct_pure_dict": self.pure_elements_dict,
                "total_target_comp": len(self.fetched_entries_dict),
            }

            with metadata_path.open("w") as json_file:
                json.dump(metadata, json_file, indent=4)

            LOGGER.info(
                f"Data saved to {output_path} and metadata saved to {metadata_path}"
            )
            return output_path, metadata_path

        except Exception as e:
            LOGGER.info(f"Error saving results: {str(e)}")
            raise

    def get_hull_refernce(
        self, formulas: Optional[List[str]] = None, verbose: bool = True
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Retrieves Material Project IDs for pure elements in their most stable phase.

        This function queries the Materials Project database for single-element materials
        that lie on the convex hull (energy_above_hull = 0), representing the most
        thermodynamically stable phase for each pure element.

        Args:
            formulas (Optional[List[str]]): List of formula strings to filter results.
                If provided, only elements matching these formulas will be returned.
                Defaults to None (all pure elements).
            verbose (bool): If True, prints details of each entry to console.
                Defaults to False.

        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: Contains two dictionaries:
                - mp_dict: Maps material_id to formula_pretty
                - name_dict: Maps formula_pretty to material_id
                If formulas is provided, only entries matching these formulas are included.
        """
        with MPRester(api_key=self.api_key) as mpr:
            results = mpr.materials.summary.search(
                num_elements=(1, 1),  # Pure elements only
                energy_above_hull=(0, 0),  # On the convex hull (most stable phase)
                fields=["material_id", "formula_pretty", "energy_above_hull"],
            )

        if verbose:
            for entry in results:
                print(
                    f"{entry.formula_pretty}: {entry.material_id} (Ehull = {entry.energy_above_hull} eV)"  # type: ignore
                )

        # If formulas are provided, filter results
        if formulas:
            filtered_results = [
                entry
                for entry in results
                if entry.formula_pretty in formulas  # type: ignore
            ]
        else:
            filtered_results = results

        mp_dict = {
            str(entry.material_id): str(entry.formula_pretty)  # type: ignore
            for entry in filtered_results
        }
        name_dict = {
            str(entry.formula_pretty): str(entry.material_id)  # type: ignore
            for entry in filtered_results
        }

        # Update pure elements dict with the mp_dict results
        self.pure_elements_dict = mp_dict

        return mp_dict, name_dict
