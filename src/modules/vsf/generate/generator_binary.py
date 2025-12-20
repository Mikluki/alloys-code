import logging
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Dict, List

from pymatgen.core import Structure
from pymatgen.util.typing import SpeciesLike

LOGGER = logging.getLogger(__name__)


class Prototypes(Enum):
    """Enum representing different crystal structure prototypes."""

    # Define enum members with tuples
    B1 = ("B1", "2x_B1__Fm3m_225_NaCl_mp-22862_poscar")
    B2 = ("B2", "2x_B2__Pm3m_221_CsCl_mp-22865_poscar")
    B3 = ("B3", "2x_B3__F43m_216_ZnS_mp-10695_poscar")
    B4 = ("B4", "2x_B4__P63mc_186_ZnO_mp-2133_poscar")
    B11 = ("B11", "2x_B11_P4nmm_129_TiCd_mp-30500_poscar")
    B19 = ("B19", "2x_B19_Pmma_51_CdAu_mp-1404poscar")
    B27 = ("B27", "2x_B27_Pnma_62_FeB_mp-20787_poscar")
    B33 = ("B33", "2x_B33_Cmcm_63_CrB_mp-260_poscar")
    L10 = ("L10", "2x_L10_P4mmm_123_CuAu_mp-522_poscar")
    L11 = ("L11", "2x_L11_R3m_166_CuPt_mp-644311_poscar")

    def __init__(self, label: str, filepath: str):
        self._value_ = label  # This is the key difference!
        self._filepath = filepath

    @property
    def label(self) -> str:
        """Return the label of the prototype."""
        return self._value_

    @property
    def filepath(self) -> str:
        """Return the filepath of the prototype."""
        return self._filepath

    def __str__(self) -> str:
        """Return the string representation of the prototype."""
        return self._value_


class GeneratorBinaryPrototype:
    def __init__(self, poscar_path: str):
        """
        Initialize the assembler with a prototype structure from a POSCAR file.

        Args:
            poscar_path: Path to the POSCAR file containing prototype structure
        """
        self.structure = Structure.from_str(
            open(poscar_path, "r", encoding="utf-8").read(), fmt="poscar"
        )
        # Get unique species from prototype for validation
        self.unique_prototype_species = list(
            set([atom.symbol for atom in self.structure.species])
        )
        if len(set(self.unique_prototype_species)) != 2:
            raise ValueError(
                f"Prototype must contain exactly 2 species. Found: {set(self.unique_prototype_species)}"
            )
        # Initialize prototype_type - will be set by from_template
        self.prototype_type = None

    @classmethod
    def from_template(cls, prototype: Prototypes) -> "GeneratorBinaryPrototype":
        """
        Create an assembler using a built-in template.

        Args:
            prototype: Type of prototype structure to use

        Returns:
            Initialized StructureAssembler
        """
        template_path = cls._get_template_path(prototype.filepath)

        # Create assembler
        assembler = cls(str(template_path))
        # Store the prototype type for later reference
        assembler.prototype_type = prototype
        return assembler

    @classmethod
    def _get_template_path(cls, template_name: str) -> Path:
        """Get path to a template file."""
        template_dir = resources.files("StructFlow.generation.templates.scaled")
        path = Path(str(template_dir.joinpath(template_name)))
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {template_name}")
        return path

    @classmethod
    def available_templates(cls) -> List[Prototypes]:
        """Returns list of available template types."""
        return list(Prototypes)

    def substitute_atoms(
        self, atom1: SpeciesLike, atom2: SpeciesLike, include_reverse: bool = False
    ) -> List[Structure]:
        """
        Create new structures by substituting atoms in the prototype structure.

        Args:
            atom1: Symbol of first element to substitute
            atom2: Symbol of second element to substitute
            include_reverse: Whether to include reverse substitution combination

        Returns:
            List of substituted structures
        """
        # Log start of substitution process
        LOGGER.info(f"Starting atom substitution with {atom1} and {atom2}")

        # Validate input atoms
        if not all(isinstance(atom, str) for atom in [atom1, atom2]):
            LOGGER.error(
                f"Type error: Atom symbols must be strings, got {type(atom1)} and {type(atom2)}"
            )
            raise TypeError("Atom symbols must be strings")

        # Create substitution mappings
        forward_map = {
            self.unique_prototype_species[0]: atom1,
            self.unique_prototype_species[1]: atom2,
        }

        # Initialize results list
        results = []

        # Create forward substitution
        LOGGER.info(
            f"Performing forward substitution: {self.unique_prototype_species[0]} → {atom1}, {self.unique_prototype_species[1]} → {atom2}"
        )
        forward_struct = self.structure.copy().replace_species(
            forward_map  # pyright: ignore
        )
        # Store element info as properties on the structure
        forward_struct.element1 = atom1  # Store composition info for later use
        forward_struct.element2 = atom2
        results.append(forward_struct)
        LOGGER.debug(
            f"Forward substitution complete, structure: {forward_struct.composition}"
        )

        # Create reverse substitution if requested
        if include_reverse:
            reverse_map = {
                self.unique_prototype_species[0]: atom2,
                self.unique_prototype_species[1]: atom1,
            }
            LOGGER.info(
                f"Performing reverse substitution: {self.unique_prototype_species[0]} → {atom2}, {self.unique_prototype_species[1]} → {atom1}"
            )
            reverse_struct = self.structure.copy().replace_species(
                reverse_map  # pyright: ignore
            )
            # Store element info as properties on the structure
            reverse_struct.element1 = atom2
            reverse_struct.element2 = atom1
            results.append(reverse_struct)
            LOGGER.debug(
                f"Reverse substitution complete, structure: {reverse_struct.composition}"
            )
        else:
            LOGGER.debug("Skipping reverse substitution (not requested)")

        LOGGER.info(f"Substitution completed, generated {len(results)} structures")
        return results

    def save_structures(
        self,
        structures: List[Structure],
        output_dir: Path | str,
        prefix: str | None = None,
    ) -> None:
        """
        Save generated structures to POSCAR files.

        Args:
            structures: List of structures to save
            output_dir: Directory to save structures in
            prefix: Optional prefix for filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for structure in structures:
            # Access element info stored as properties on the structure
            elem1 = getattr(structure, "element1", "Unknown")
            elem2 = getattr(structure, "element2", "Unknown")
            filename = f"{prefix + '_' if prefix else ''}{elem1}_{elem2}_POSCAR"
            structure.to(filename=str(output_path / filename), fmt="poscar")

    @property
    def composition(self) -> Dict[str, float]:
        """Returns composition of prototype structure"""
        return self.structure.composition.as_dict()

    def get_wyckoff_positions(self) -> Dict:
        """Returns Wyckoff positions of the prototype structure"""
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        analyzer = SpacegroupAnalyzer(self.structure)
        return analyzer.get_symmetry_dataset()["wyckoffs"]

    @staticmethod
    def count_ordered_pairs(n_elements: int) -> int:
        """
        Estimate number of ordered pairs without repeats (no X-X) given number of elements.

        Args:
            n_elements (int): Number of unique elements.

        Returns:
            int: Number of ordered pairs.
        """
        if n_elements < 2:
            return 0  # You need at least 2 elements to form a pair
        return n_elements * (n_elements - 1)


if __name__ == "__main__":
    # Create assembler with a specific prototype
    prototype = Prototypes.B2
    assembler = GeneratorBinaryPrototype.from_template(prototype)

    # Get the prototype name
    name = prototype.label  # Returns "B2"
    # Or using str()
    name_str = str(prototype)  # Returns "B2"

    # Get the filepath
    filepath = prototype.filepath

    print(f"Prototype name: {name}")
    print(f"Prototype string: {name_str}")
    print(f"Prototype filepath: {filepath}")
