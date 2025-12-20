import logging
import shutil
from pathlib import Path
from typing import Dict, List

LOGGER = logging.getLogger(__name__)


class IdGenerator:
    """Generates unique IDs with collision avoidance. Pure ID logic with no side effects."""

    def __init__(self, starting_id: int):
        """
        Initialize generator with explicit starting ID.

        Parameters:
        -----------
        starting_id : int
            Starting ID value (e.g., 1000000 for catalysis, 2000000 for random)
        """
        self._starting_id = starting_id

    def _find_next_available_id(self, target_dir: Path, prefix: str) -> int:
        """
        Find the next available ID by scanning existing directories.

        Parameters:
        -----------
        target_dir : Path
            Directory to scan for existing structures
        prefix : str
            Prefix to search for (e.g., "catalysis")

        Returns:
        --------
        int
            Next available ID for this prefix
        """
        if not target_dir.exists():
            return self._starting_id

        existing_ids = []
        pattern = f"{prefix}_*"

        for dir_path in target_dir.glob(pattern):
            if dir_path.is_dir():
                try:
                    # Extract numeric part after prefix_
                    id_part = dir_path.name.split(f"{prefix}_")[1]
                    existing_ids.append(int(id_part))
                except (IndexError, ValueError):
                    # Skip malformed directory names
                    continue

        if existing_ids:
            return max(existing_ids) + 1
        else:
            return self._starting_id

    def peek_next_id(self, target_dir: Path, prefix: str) -> int:
        """
        Get next available ID without any side effects.

        Parameters:
        -----------
        target_dir : Path
            Directory to check for existing structures
        prefix : str
            Prefix to search for

        Returns:
        --------
        int
            Next available ID that would be assigned
        """
        return self._find_next_available_id(target_dir, prefix)

    def peek_next_batch(self, target_dir: Path, prefix: str, count: int) -> List[str]:
        """
        Get next N structure IDs for batch without side effects.

        Parameters:
        -----------
        target_dir : Path
            Directory to check for existing structures
        prefix : str
            Prefix for structure IDs
        count : int
            Number of IDs to generate

        Returns:
        --------
        List[str]
            List of structure IDs that would be assigned
            ["catalysis_1000001", "catalysis_1000002", ...]
        """
        next_id = self._find_next_available_id(target_dir, prefix)
        return [f"{prefix}_{next_id + i}" for i in range(count)]

    def generate_batch_ids(
        self, target_dir: Path, prefix: str, count: int
    ) -> List[str]:
        """
        Generate batch of IDs for actual use.

        Parameters:
        -----------
        target_dir : Path
            Directory where structures will be organized
        prefix : str
            Prefix for structure IDs
        count : int
            Number of IDs to generate

        Returns:
        --------
        List[str]
            List of structure IDs to use
        """
        # For now, same as peek_next_batch since we're stateless
        # Could add logging or other behavior here if needed
        batch_ids = self.peek_next_batch(target_dir, prefix, count)
        LOGGER.info(
            f"Generated {count} IDs for prefix '{prefix}': {batch_ids[0]} to {batch_ids[-1]}"
        )
        return batch_ids


class PoscarOrganizer:
    """Takes flat list of POSCAR files and copies each into its own directory with generated ID."""

    def __init__(self, id_generator: IdGenerator):
        """
        Parameters:
        -----------
        id_generator : IdGenerator
            Object that generates sequential IDs and handles collision avoidance
        """
        self._id_generator = id_generator
        self.starting_id = self._id_generator._starting_id

    @classmethod
    def from_starting_id(cls, starting_id: int):
        """
        Convenience constructor that creates an internal IdGenerator.

        Parameters:
        -----------
        starting_id : int
            Starting ID value (e.g., 1000000 for catalysis, 2000000 for random)

        Returns:
        --------
        PoscarOrganizer
            Organizer instance with internal IdGenerator
        """
        return cls(IdGenerator(starting_id))

    def organize_poscar_list(
        self, poscar_paths: List[Path], target_dir: Path, prefix: str
    ) -> Dict[str, Path]:
        """
        Organize POSCAR files into individual directories with sequential IDs.

        Parameters:
        -----------
        poscar_paths : List[Path]
            List of POSCAR file paths to organize
        target_dir : Path
            Where to create organized structure
        prefix : str
            Semantic prefix for IDs (e.g., "catalysis", "random")

        Returns:
        --------
        Dict[str, Path]
            Mapping from structure_id -> organized_directory_path
            {"catalysis_1000001": Path("target/catalysis_1000001"), ...}
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Generate all IDs upfront
        struct_ids = self._id_generator.generate_batch_ids(
            target_dir, prefix, len(poscar_paths)
        )

        organized_dirs = {}

        for poscar_path, struct_id in zip(poscar_paths, struct_ids):
            # Create target directory
            struct_dir = target_dir / struct_id
            struct_dir.mkdir(exist_ok=True)

            # Copy POSCAR to organized location
            target_poscar = struct_dir / "POSCAR"
            shutil.copy2(poscar_path, target_poscar)

            organized_dirs[struct_id] = struct_dir

            LOGGER.debug(f"Organized {poscar_path} -> {struct_dir}")

        return organized_dirs


def test_organizer_workflow():
    """Test IdGenerator and PoscarOrganizer with 5 POSCARs, rand prefix, starting_id=2000000."""
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    # Create test directory in cwd
    test_dir = Path.cwd() / "organizer_test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    try:
        # Create 5 test POSCAR files with different structures
        poscar_paths = []

        for i in range(5):
            # Create simple test structures with slight variations
            atoms = Atoms(
                symbols=["C", "C"],
                positions=[[0.1 + i * 0.1, 0.1, 0.1], [1.4, 1.4 + i * 0.1, 1.4]],
                cell=[3.0 + i * 0.1, 3.0, 3.0],
                pbc=True,
            )

            # Save as POSCAR file
            struct = AseAtomsAdaptor.get_structure(atoms)  # pyright: ignore
            poscar_path = test_dir / f"test_poscar_{i}.vasp"
            struct.to(filename=str(poscar_path), fmt="poscar")
            poscar_paths.append(poscar_path)

        print(f"Created 5 test POSCAR files in {test_dir}")

        # Test dry run capability
        id_gen = IdGenerator(starting_id=2000000)
        target_dir = test_dir / "organized"

        print("\n=== Dry Run ===")
        next_id = id_gen.peek_next_id(target_dir, "rand")
        print(f"Next available ID: rand_{next_id}")

        next_batch = id_gen.peek_next_batch(target_dir, "rand", count=5)
        print(f"Next batch of 5 IDs: {next_batch}")

        # Actual organization
        print("\n=== Actual Organization ===")
        organizer = PoscarOrganizer(id_gen)
        organized_dirs = organizer.organize_poscar_list(
            poscar_paths, target_dir, "rand"
        )

        print(f"Organized {len(organized_dirs)} structures:")
        for struct_id, struct_dir in organized_dirs.items():
            print(f"  {struct_id} -> {struct_dir}")

        print(f"\nTest completed successfully!")
        print(f"Results available at: {target_dir}")

        # Test collision avoidance - run again
        print("\n=== Testing Collision Avoidance ===")
        next_id_after = id_gen.peek_next_id(target_dir, "rand")
        print(f"Next ID after organization: rand_{next_id_after}")

    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    test_organizer_workflow()
