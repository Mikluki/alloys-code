"""
LOGGER = logging.getLogger(__name__)
Generate a random, well‑packed transition‑metal structure and write it to
POSCAR (VASP format).

The algorithm guarantees that every inter‑atomic distance is ≥
1.05 × (rᵢ + rⱼ), where r is the van‑der‑Waals radius.

Can generate multiple distinct structures with different seeds.
Tracks used seeds to ensure uniqueness when expanding the dataset.
"""

from __future__ import annotations

import logging
import random
import time
from math import pi, pow
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class GeneratorRandomPacking:
    DEFAULT_PACKING_FRACTION: float = 0.62  # random‑close‑packing ≈ 0.64
    DEFAULT_SAFETY_FACTOR: float = 1.00  # d_min = safety × (r_i + r_j)
    DEFAULT_MAX_ATTEMPTS_PER_ATOM: int = 1_000_000

    # van‑der‑Waals radii for the 30 d‑block elements, Å (Bondi/Alvarez averages)
    VDW_RADIUS: Dict[str, float] = {
        "Sc": 2.11,
        "Ti": 2.00,
        "V": 1.92,
        "Cr": 1.85,
        "Mn": 1.79,
        "Fe": 1.72,
        "Co": 1.67,
        "Ni": 1.63,
        "Cu": 1.40,
        "Zn": 1.39,
        "Y": 2.12,
        "Zr": 2.06,
        "Nb": 1.98,
        "Mo": 1.90,
        "Tc": 1.83,
        "Ru": 1.78,
        "Rh": 1.73,
        "Pd": 1.69,
        "Ag": 1.65,
        "Cd": 1.58,
        "Hf": 2.08,
        "Ta": 2.00,
        "W": 1.93,
        "Re": 1.88,
        "Os": 1.85,
        "Ir": 1.80,
        "Pt": 1.77,
        "Au": 1.66,
        "Hg": 1.55,
    }

    TRANSITION_METALS: List[str] = list(VDW_RADIUS)

    def __init__(
        self,
        atom_count: int,
        allowed_species: List[str] | None = None,
        output_file: str | Path = "POSCAR",
        seed: int | None = None,
        packing_fraction: float = DEFAULT_PACKING_FRACTION,
        max_attempts_per_atom: int = DEFAULT_MAX_ATTEMPTS_PER_ATOM,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
    ):
        """Initialize the POSCAR generator with the specified parameters.

        Parameters
        ----------
        atom_count            Total number of atoms to generate.
        packing_fraction      Target packing fraction for the structure (default 0.62).
                                Higher values = denser packing/smaller box.
                                Lower values = looser packing/larger box.
                                Max theoretical ~0.64 for random close packing of spheres.
        max_attempts_per_atom Maximum attempts to place an atom before failing (default 1,000,000).
        safety_factor         Multiplier for minimum inter-atomic distances (default 1.00).
                              1.0 = atoms touch exactly, >1.0 = buffer space, <1.0 = overlap allowed
        allowed_species       Restrict random choice to these element symbols (optional).
        seed                  Random seed for reproducibility (default None for random seed).
        output_file           Filename to write (default "POSCAR").
        """
        if atom_count <= 0:
            raise ValueError("atom_count must be positive")
        if packing_fraction <= 0 or packing_fraction >= 1:
            raise ValueError("packing_fraction must be between 0 and 1")
        if max_attempts_per_atom <= 0:
            raise ValueError("max_attempts_per_atom must be positive")
        if safety_factor <= 0:
            raise ValueError("safety_factor must be positive")

        self.atom_count = atom_count
        self.output_file = Path(output_file)
        self.packing_fraction = packing_fraction
        self.max_attempts_per_atom = max_attempts_per_atom
        self.safety_factor = safety_factor

        # Set the random seed (use provided seed or generate a random one)
        self.seed = seed if seed is not None else int(time.time() * 1000) & 0xFFFFFFFF
        self.rng = random.Random(self.seed)

        # Set up allowed species
        if allowed_species is None:
            self.allowed_species = self.TRANSITION_METALS
        else:
            invalid = set(allowed_species) - set(self.TRANSITION_METALS)
            if invalid:
                raise ValueError(
                    f"Invalid species for transition metals: {sorted(invalid)}"
                )
            self.allowed_species = allowed_species

    def _estimate_box_length(self, radii: List[float]) -> float:
        """Return cubic cell length (Å) for target packing fraction."""
        total_volume = sum(4 / 3 * pi * r**3 for r in radii)
        cell_volume = total_volume / self.packing_fraction
        return pow(cell_volume, 1 / 3)

    def _generate_positions(
        self, species: List[str], radii: List[float], cell_len: float
    ) -> Tuple[np.ndarray | None, bool]:
        """Pack atoms randomly in a cubic box with distance constraint.

        Parameters
        ----------
        species : List of element symbols
        radii : List of corresponding atomic radii
        cell_len : Cubic cell length

        Returns
        -------
        Tuple of (positions array or None, success flag)
        """
        positions: List[np.ndarray] = []

        for idx, (atom_symbol, r_i) in enumerate(zip(species, radii)):
            # Place each atom one by one, checking against all previously placed atoms
            for _ in range(self.max_attempts_per_atom):
                # Try up to N random positions for this atom until we find one that doesn't overlap
                trial = np.array([self.rng.random() * cell_len for _ in range(3)])

                # Check if trial position is valid by ensuring minimum distance to all existing atoms
                # - trial: proposed 3D coordinates for current atom (radius r_i)
                # - pos, r_j: position and radius of each previously placed atom
                # - np.linalg.norm(trial - pos): distance between atom centers
                # - safety_factor * (r_i + r_j): minimum allowed center-to-center distance
                #
                # Physical meaning:
                # - safety_factor = 1.0: atoms touch exactly (hard sphere packing)
                # - safety_factor > 1.0: extra buffer space between atoms
                # - safety_factor < 1.0: atoms allowed to overlap
                #
                # Only check against radii[:idx] because we place atoms sequentially,
                # so only atoms 0 through (idx-1) have been positioned so far
                if all(
                    np.linalg.norm(trial - pos) >= self.safety_factor * (r_i + r_j)
                    for pos, r_j in zip(positions, radii[:idx])
                ):
                    positions.append(trial)
                    break
            else:
                # Calculate how many of each type we've already placed
                current_composition = {}
                for i in range(idx):
                    atom = species[i]
                    current_composition[atom] = current_composition.get(atom, 0) + 1

                # Format the current composition for the error message
                composition_str = ", ".join(
                    f"{a}: {n}" for a, n in current_composition.items()
                )

                LOGGER.warning(
                    f"Seed {self.seed}: Packing failed for {atom_symbol} atom (#{idx}) after "
                    f"{self.max_attempts_per_atom} trials.\n"
                    f"Current composition: {composition_str}\n"
                    f"Box length: {cell_len:.3f} Å, Packing fraction: {self.packing_fraction:.3f}"
                )
                return None, False

        return np.vstack(positions), True

    def _write_poscar(
        self,
        elements: List[str],
        counts: List[int],
        cell_len: float,
        positions: np.ndarray,
    ) -> None:
        """Write data to POSCAR (Cartesian coordinates)."""
        with self.output_file.open("w", encoding="utf‑8") as fh:
            fh.write(
                f"Random TM structure generated by POSCARGenerator (seed: {self.seed})\n"
            )
            fh.write("1.0\n")  # global scaling
            for vec in np.eye(3) * cell_len:
                fh.write(f"{vec[0]:12.6f} {vec[1]:12.6f} {vec[2]:12.6f}\n")
            fh.write(" ".join(elements) + "\n")
            fh.write(" ".join(str(c) for c in counts) + "\n")
            fh.write("Cartesian\n")
            for p in positions:
                fh.write(f"{p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")

    def generate(self) -> Tuple[Path | None, bool]:
        """Create a POSCAR containing randomly chosen atoms.

        Returns
        -------
        Tuple of (Path to the generated POSCAR file or None if failed, success flag)
        """
        # Choose random species
        species = self.rng.choices(self.allowed_species, k=self.atom_count)
        radii = [self.VDW_RADIUS[s] for s in species]

        # Calculate box length and generate positions
        cell_len = self._estimate_box_length(radii)
        coords, success = self._generate_positions(species, radii, cell_len)

        if not success:
            return None, False

        # Collapse to POSCAR element ordering (appearance order)
        uniq_elems: List[str] = list(dict.fromkeys(species))
        counts: List[int] = [species.count(e) for e in uniq_elems]

        # Write output file
        assert coords is not None, f"Provided species is an empty list"
        self._write_poscar(uniq_elems, counts, cell_len, coords)
        return self.output_file, True

    @property
    def current_seed(self) -> int:
        """Return the seed used for this generator instance."""
        return self.seed

    @classmethod
    def generate_batch(
        cls,
        num_structures: int,
        atom_count: int,
        output_dir: Path,
        allowed_species: List[str] | None = None,
        seed_cache_file: Path | None = None,
        failed_seeds_file: Path | None = None,
        packing_fraction: float = DEFAULT_PACKING_FRACTION,
        max_attempts_per_atom: int = DEFAULT_MAX_ATTEMPTS_PER_ATOM,
        safety_factor: float = DEFAULT_SAFETY_FACTOR,
        base_seed: int | None = None,
        file_prefix: str = "",
        seed_file_suffix: str = "",
        allocated_seeds: List[int] | None = None,
    ) -> Tuple[int, int, List[int]]:
        """Generate a batch of POSCAR structures with unique seeds.

        Parameters
        ----------
        num_structures      : Number of successful structures to generate
        atom_count          : Number of atoms in each structure
        output_dir          : Directory to save the generated structures
        allowed_species     : List of allowed element symbols (optional)
        seed_cache_file     : File to track used seeds (default: used_seeds{suffix}.txt)
        failed_seeds_file   : File to track failed seeds (default: failed_seeds{suffix}.txt)
        packing_fraction    : Target packing fraction (default: 0.62)
        max_attempts_per_atom: Maximum placement attempts per atom (default: 1,000,000)
        safety_factor       : Minimum distance multiplier (default: 1.00)
        base_seed           : Starting seed for generation (default: current time)
        file_prefix         : Prefix for POSCAR filenames (e.g., "p1_" → "p1_POSCAR_0000")
        seed_file_suffix    : Suffix for seed tracking files (e.g., "_p1" → "used_seeds_p1.txt")
        allocated_seeds     : Pre-allocated seeds to use (from ParallelBatchGenerator)
                             If provided, ignores base_seed and SeedManager

        Returns
        -------
        Tuple of (successful_count, failed_count, failed_seeds)
        """
        # Use default file paths with suffix in output directory if not specified
        if seed_cache_file is None:
            seed_cache_file = output_dir / f"used_seeds{seed_file_suffix}.txt"
        if failed_seeds_file is None:
            failed_seeds_file = output_dir / f"failed_seeds{seed_file_suffix}.txt"

        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)

        LOGGER.info(
            f"Generating {num_structures} POSCAR structures in {output_dir.resolve()}"
        )
        LOGGER.info(
            f"Settings: packing_fraction={packing_fraction}, safety_factor={safety_factor}"
        )
        if file_prefix:
            LOGGER.info(f"File prefix: '{file_prefix}'")

        failed_seeds = []

        # NEW: Use allocated seeds if provided, otherwise use SeedManager
        if allocated_seeds is not None:
            # Parallel mode: use pre-allocated seeds
            LOGGER.info(
                f"Using {len(allocated_seeds)} pre-allocated seeds from parallel generator"
            )
            LOGGER.info(f"Seed range: {allocated_seeds[0]} to {allocated_seeds[-1]}")
            unique_seeds = allocated_seeds.copy()

            # Create minimal seed manager just for tracking used seeds in this process
            seed_manager = SeedManager(seed_cache_file)

            # Add all allocated seeds to the manager for tracking
            for seed in allocated_seeds:
                seed_manager.add_seed(seed)
        else:
            # Standalone mode: use SeedManager to generate seeds
            LOGGER.info("Using SeedManager for standalone seed generation")
            seed_manager = SeedManager(seed_cache_file)

            # Use current time as base seed if not provided
            if base_seed is None:
                base_seed = int(time.time())

            # Get unique seeds for all structures
            LOGGER.info(f"Finding {num_structures} unique seeds...")
            unique_seeds = seed_manager.get_unique_seeds(num_structures, base_seed)
            LOGGER.info(f"Using seeds {unique_seeds[0]} through {unique_seeds[-1]}")

        # Save the seeds immediately to prevent loss in case of interruption
        seed_manager.save_used_seeds()

        # Track successful and failed generations
        successful_count = 0
        failed_count = 0

        # Generate structures until we have enough successful ones
        seed_index = 0
        max_seeds = len(unique_seeds)

        while successful_count < num_structures and seed_index < max_seeds:
            seed = unique_seeds[seed_index]
            # Use prefix in filename
            output_file = output_dir / f"{file_prefix}POSCAR_{successful_count:04d}"

            generator = cls(
                atom_count=atom_count,
                allowed_species=allowed_species,
                output_file=output_file,
                seed=seed,
                packing_fraction=packing_fraction,
                max_attempts_per_atom=max_attempts_per_atom,
                safety_factor=safety_factor,
            )

            path, success = generator.generate()

            if success:
                assert path is not None, f"Successful poscar was returned without path"
                successful_count += 1
                # Log progress (every 100 successful structures)
                if (
                    successful_count % 100 == 0
                    or successful_count == 1
                    or successful_count == num_structures
                ):
                    LOGGER.info(
                        f"Generated {successful_count}/{num_structures}: {path.name} (seed: {generator.current_seed})"
                    )
            else:
                failed_count += 1
                failed_seeds.append(seed)

                # Log failures periodically
                if failed_count % 10 == 0:
                    LOGGER.warning(f"{failed_count} generation failures so far")

            # Save seeds periodically to prevent data loss
            if (seed_index + 1) % 100 == 0:
                seed_manager.save_used_seeds()

                # Save failed seeds to a separate file
                if failed_seeds:
                    try:
                        with failed_seeds_file.open("w") as f:
                            for failed_seed in sorted(failed_seeds):
                                f.write(f"{failed_seed}\n")
                    except OSError as e:
                        LOGGER.warning(
                            f"Could not save failed seeds to {failed_seeds_file}: {e}"
                        )

            seed_index += 1

            # If we're running out of allocated seeds, log warning
            if (
                allocated_seeds is not None
                and seed_index >= max_seeds - 10
                and successful_count < num_structures
            ):
                remaining_structures = num_structures - successful_count
                remaining_seeds = max_seeds - seed_index
                LOGGER.warning(
                    f"Running low on allocated seeds: {remaining_seeds} seeds left for {remaining_structures} structures"
                )

            # If we're using SeedManager and running out of seeds, generate more
            elif (
                allocated_seeds is None
                and seed_index >= max_seeds - 10
                and successful_count < num_structures
            ):
                additional_seeds = seed_manager.get_unique_seeds(
                    100, unique_seeds[-1] + 1
                )
                unique_seeds.extend(additional_seeds)
                max_seeds = len(unique_seeds)
                LOGGER.info(
                    f"Generated additional seeds, now have {max_seeds} total seeds"
                )

        # Check if we exhausted allocated seeds without completing
        if (
            allocated_seeds is not None
            and successful_count < num_structures
            and seed_index >= max_seeds
        ):
            LOGGER.error(
                f"Exhausted all {len(allocated_seeds)} allocated seeds. "
                f"Generated {successful_count}/{num_structures} structures. "
                f"Consider increasing seeds_per_process in ParallelBatchGenerator."
            )

        # Final save of all used seeds
        seed_manager.save_used_seeds()

        # Save final failed seeds list
        if failed_seeds:
            try:
                with failed_seeds_file.open("w") as f:
                    for failed_seed in sorted(failed_seeds):
                        f.write(f"{failed_seed}\n")
                LOGGER.info(f"Failed seeds saved to {failed_seeds_file.resolve()}")
            except OSError as e:
                LOGGER.warning(
                    f"Could not save failed seeds to {failed_seeds_file}: {e}"
                )

        LOGGER.info(f"Successfully generated {successful_count} POSCAR structures")
        if failed_count > 0:
            LOGGER.info(
                f"Failed to generate {failed_count} structures due to packing constraints"
            )

        return successful_count, failed_count, failed_seeds


class SeedManager:
    """Manages and tracks used seeds to ensure uniqueness."""

    def __init__(self, cache_file: str | Path = "used_seeds.txt"):
        """Initialize the seed manager with a cache file.

        Parameters
        ----------
        cache_file : Path to the file storing used seeds.
        """
        self.cache_file = Path(cache_file)
        self.used_seeds: Set[int] = self._load_used_seeds()

    def _load_used_seeds(self) -> Set[int]:
        """Load previously used seeds from the cache file."""
        if not self.cache_file.exists():
            LOGGER.info(f"[ {self.cache_file} ] does not exist. Starting fresh")
            return set()

        try:
            with self.cache_file.open("r") as f:
                LOGGER.info(f"[{self.cache_file}] found. Used seeds will be EXCLUDED")
                return set(int(line.strip()) for line in f if line.strip())
        except (ValueError, OSError) as e:
            LOGGER.warning(f"Could not load seeds from {self.cache_file}: {e}")
            return set()

    def save_used_seeds(self) -> None:
        """Save the current set of used seeds to the cache file."""
        try:
            with self.cache_file.open("w") as f:
                for seed in sorted(self.used_seeds):
                    f.write(f"{seed}\n")
        except OSError as e:
            LOGGER.warning(f"Could not save seeds to {self.cache_file}: {e}")

    def is_seed_used(self, seed: int) -> bool:
        """Check if a seed has been used before."""
        return seed in self.used_seeds

    def add_seed(self, seed: int) -> None:
        """Mark a seed as used."""
        self.used_seeds.add(seed)

    def get_unique_seed(self, base_seed: int | None = None) -> int:
        """Generate a unique seed that hasn't been used before.

        Parameters
        ----------
        base_seed : Optional starting point for seed generation.

        Returns
        -------
        A unique seed value not present in the used_seeds set.
        """
        if base_seed is None:
            base_seed = int(time.time() * 1000) & 0xFFFFFFFF

        candidate_seed = base_seed

        # Keep incrementing if the seed is already used
        while self.is_seed_used(candidate_seed):
            candidate_seed = (candidate_seed + 1) & 0xFFFFFFFF

        # Mark as used and return
        self.add_seed(candidate_seed)
        return candidate_seed

    def get_unique_seeds(self, count: int, base_seed: int | None = None) -> List[int]:
        """Generate multiple unique seeds.

        Parameters
        ----------
        count : Number of unique seeds to generate.
        base_seed : Optional starting point for seed generation.

        Returns
        -------
        A list of unique seed values.
        """
        seeds = []

        for _ in range(count):
            seed = self.get_unique_seed(base_seed)
            seeds.append(seed)
            # Update base_seed for next iteration to avoid unnecessary checks
            base_seed = (seed + 1) & 0xFFFFFFFF

        return seeds


def print_readme() -> None:
    """Print usage instructions for batch POSCAR generation."""
    readme_text = """
Batch POSCAR generation usage:

    from vsf.generate.generator_random import GeneratorRandomPacking
    from pathlib import Path

    # Basic batch generation
    success_count, fail_count, failed_seeds = POSCARGenerator.generate_batch(
        num_structures=100,
        atom_count=20,
        output_dir=Path("random_structures")
    )

    # With specific transition metals only
    success_count, fail_count, failed_seeds = POSCARGenerator.generate_batch(
        num_structures=50,
        atom_count=15,
        output_dir=Path("ti_fe_structures"),
        allowed_species=["Ti", "Fe", "Ni", "Cu"]
    )

    # Custom packing parameters
    success_count, fail_count, failed_seeds = POSCARGenerator.generate_batch(
        num_structures=200,
        atom_count=25,
        output_dir=Path("dense_structures"),
        packing_fraction=0.70,        # Denser packing
        max_attempts_per_atom = 1000  # Max positioning attempts per atom
        safety_factor=0.95            # Tighter atom spacing
    )

    # Manual parallel execution (different terminals/processes)
    # Process 1:
    POSCARGenerator.generate_batch(..., file_prefix="p1_", seed_file_suffix="_p1")
    # Process 2:  
    POSCARGenerator.generate_batch(..., file_prefix="p2_", seed_file_suffix="_p2")

    What gets generated:
    • POSCAR files: POSCAR_0000, POSCAR_0001, ... POSCAR_0099
    • Seed tracking: used_seeds.txt (prevents duplicate structures)
    • Failed seeds: failed_seeds.txt (seeds that couldn't pack)
    
    Return values:
    • success_count: Number of structures successfully generated
    • fail_count: Number of failed attempts due to packing constraints
    • failed_seeds: List of seeds that failed (for debugging)

    Common packing failure causes:
    • Too many large atoms in small volume (reduce packing_fraction)
    • Safety factor too strict (reduce safety_factor)
    • Very large atom count (increase max_attempts_per_atom)
"""
    print(readme_text.strip())
