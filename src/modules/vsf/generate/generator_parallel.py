import logging
import multiprocessing as mp
import pprint
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

LOGGER = logging.getLogger(__name__)


class ParallelBatchGenerator:
    """Generic parallel batch generator for structure generation classes."""

    def __init__(
        self,
        generator_class: Type,
        num_structures: int,
        num_processes: int,
        output_dir: Path,
        existing_used_seeds: Path | set[int] | None | None = None,
        seeds_per_process: int = 10_000,  # NEW: explicit, configurable
    ):
        """
        Initialize parallel batch generator.

        Parameters
        ----------
        generator_class : Type
            Class with generate_batch() method (e.g., GeneratorRandomPacking)
        num_structures : int
            Total number of structures to generate across all processes
        num_processes : int
            Number of parallel processes to spawn
        output_dir : Path
            Directory for output (each process gets a subdirectory)
        existing_used_seeds : Path | Set[int] | None
            Previously used seeds to exclude from allocation.
            Can be a path to used_seeds.txt file or a set of integers.
        seeds_per_process : int
            Number of seeds to allocate per process (default: 10,000)
        """
        if num_structures <= 0:
            raise ValueError("num_structures must be positive")
        if num_processes <= 0:
            raise ValueError("num_processes must be positive")
        if num_processes > num_structures:
            raise ValueError("num_processes cannot exceed num_structures")
        if seeds_per_process <= 0:
            raise ValueError("seeds_per_process must be positive")

        self.generator_class = generator_class
        self.num_structures = num_structures
        self.num_processes = num_processes
        self.output_dir = Path(output_dir)
        self.existing_used_seeds = existing_used_seeds
        self.seeds_per_process = seeds_per_process

        # Calculate work distribution
        self.structures_per_process = num_structures // num_processes
        self.extra_structures = num_structures % num_processes

    def _load_used_seeds(self) -> set[int]:
        """Load existing used seeds from file or set."""
        used_seeds = set()

        if self.existing_used_seeds is None:
            return used_seeds

        if isinstance(self.existing_used_seeds, Path):
            # Load from file
            if self.existing_used_seeds.exists():
                try:
                    with self.existing_used_seeds.open("r") as f:
                        used_seeds = set(
                            int(line.strip()) for line in f if line.strip()
                        )
                    LOGGER.info(
                        f"Loaded {len(used_seeds)} used seeds FROM [ {self.existing_used_seeds}]"
                    )
                except (ValueError, OSError) as e:
                    LOGGER.warning(
                        f"Could not load seeds from {self.existing_used_seeds}: {e}"
                    )
            else:
                LOGGER.info(
                    f"Used seeds file {self.existing_used_seeds} does not exist - starting fresh"
                )
        elif isinstance(self.existing_used_seeds, set):
            used_seeds = self.existing_used_seeds.copy()
            LOGGER.info(f"Using provided set of {len(used_seeds)} used seeds")
        else:
            LOGGER.warning(
                f"Invalid existing_used_seeds type: {type(self.existing_used_seeds)}"
            )

        return used_seeds

    def _allocate_seeds_for_processes(self) -> List[List[int]]:
        """Allocate non-overlapping seed ranges for each process."""
        used_seeds = self._load_used_seeds()

        # Allocate seeds for each process
        all_process_seeds = []
        current_seed = 1  # Start from 1

        LOGGER.info(
            f"Allocating {self.seeds_per_process} seeds per process for {self.num_processes} processes"
        )
        LOGGER.info(f"Excluding {len(used_seeds)} previously used seeds")

        for process_id in range(self.num_processes):
            process_seeds = []
            seeds_allocated = 0

            while seeds_allocated < self.seeds_per_process:
                # Skip used seeds
                while current_seed in used_seeds:
                    current_seed += 1

                process_seeds.append(current_seed)
                used_seeds.add(current_seed)  # Mark as allocated to prevent overlap
                current_seed += 1
                seeds_allocated += 1

            all_process_seeds.append(process_seeds)
            LOGGER.info(
                f"Process {process_id}: allocated seeds {process_seeds[0]} to {process_seeds[-1]}"
            )

        return all_process_seeds

    def _get_process_config(
        self, process_id: int, allocated_seeds: List[int]
    ) -> Dict[str, Any]:
        """Get configuration for a specific process."""
        # Calculate structures for this process
        structures_for_this_process = self.structures_per_process
        if process_id < self.extra_structures:
            structures_for_this_process += 1

        # Create process-specific directory and naming
        process_dir = self.output_dir / f"process_{process_id:02d}"
        file_prefix = f"p{process_id:02d}_"
        seed_suffix = f"_p{process_id:02d}"

        return {
            "num_structures": structures_for_this_process,
            "output_dir": process_dir,
            "file_prefix": file_prefix,
            "seed_file_suffix": seed_suffix,
            "allocated_seeds": allocated_seeds,  # NEW: pass allocated seeds
            "process_id": process_id,
        }

    def _run_single_process(
        self, full_config: Dict[str, Any]
    ) -> Tuple[int, int, int, List[int]]:
        """Run generation for a single process."""
        process_id = full_config.pop("process_id")

        LOGGER.info(
            f"Process {process_id}: Starting generation of {full_config['num_structures']} structures"
        )

        try:
            # Call the generator's batch method with allocated seeds
            success_count, fail_count, failed_seeds = (
                self.generator_class.generate_batch(**full_config)
            )

            LOGGER.info(
                f"Process {process_id}: Completed - {success_count} success, {fail_count} failed"
            )
            return process_id, success_count, fail_count, failed_seeds

        except Exception as e:
            LOGGER.error(f"Process {process_id}: Failed with error: {e}")
            return process_id, 0, 0, []

    def generate(self, **generator_kwargs) -> Dict[str, Any]:
        """
        Generate structures in parallel.

        Parameters
        ----------
        **generator_kwargs
            Additional arguments to pass to the generator's generate_batch method

        Returns
        -------
        Dict containing:
            - total_success: Total successful structures across all processes
            - total_failed: Total failed attempts across all processes
            - process_results: List of results from each process
            - failed_seeds: Combined list of all failed seeds
            - output_dirs: List of process output directories
            - all_allocated_seeds: All seeds allocated across processes (for saving)
        """
        LOGGER.info(
            f"Starting parallel generation: {self.num_structures} structures across {self.num_processes} processes"
        )
        LOGGER.info(f"Seeds per process: {self.seeds_per_process}")

        # Create main output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Allocate seeds for all processes
        all_process_seeds = self._allocate_seeds_for_processes()

        # Prepare process configurations
        process_configs = []
        for i in range(self.num_processes):
            config = self._get_process_config(i, all_process_seeds[i])
            # Merge config and generator_kwargs
            full_config = {**config, **generator_kwargs}
            process_configs.append(full_config)

        # Run processes in parallel
        with mp.Pool(processes=self.num_processes) as pool:
            results = pool.map(self._run_single_process, process_configs)

        # Collect results
        total_success = 0
        total_failed = 0
        all_failed_seeds = []
        process_results = []
        output_dirs = []
        all_allocated_seeds = []

        for process_id, success, failed, failed_seeds in results:
            total_success += success
            total_failed += failed
            all_failed_seeds.extend(failed_seeds)
            all_allocated_seeds.extend(all_process_seeds[process_id])
            process_results.append(
                {
                    "process_id": process_id,
                    "success_count": success,
                    "fail_count": failed,
                    "failed_seeds": failed_seeds,
                }
            )
            output_dirs.append(self.output_dir / f"process_{process_id:02d}")

        LOGGER.info(
            f"Parallel generation complete: {total_success}/{self.num_structures} successful"
        )

        return {
            "total_success": total_success,
            "total_failed": total_failed,
            "process_results": process_results,
            "failed_seeds": all_failed_seeds,
            "output_dirs": output_dirs,
            "all_allocated_seeds": all_allocated_seeds,  # NEW: for saving to used_seeds.txt
        }

    def save_all_used_seeds(
        self, results: Dict[str, Any], output_file: Path | None = None
    ) -> None:
        """
        Save all allocated seeds to a consolidated used_seeds.txt file.

        Parameters
        ----------
        results : Dict
            Results from generate() method
        output_file : Path, optional
            Path to save used seeds (default: output_dir/used_seeds.txt)
        """
        if output_file is None:
            output_file = self.output_dir / "used_seeds.txt"

        try:
            with output_file.open("w") as f:
                for seed in sorted(results["all_allocated_seeds"]):
                    f.write(f"{seed}\n")
            LOGGER.info(
                f"Saved {len(results['all_allocated_seeds'])} allocated seeds to {output_file}"
            )
        except OSError as e:
            LOGGER.error(f"Could not save seeds to {output_file}: {e}")


def print_readme() -> None:
    """Print usage instructions for parallel batch generation."""
    readme_text = """
Parallel batch generation usage:

    from vsf.generate.generator_parallel import ParallelBatchGenerator
    from vsf.generate.generator_random import GeneratorRandomPacking
    from pathlib import Path

    # Basic parallel generation (fresh start)
    parallel_gen = ParallelBatchGenerator(
        generator_class=GeneratorRandomPacking,
        num_structures=500,
        num_processes=5,
        output_dir=Path("parallel_structures"),
        seeds_per_process=10_000  # Explicit seed allocation
    )

    results = parallel_gen.generate(
        atom_count=20,
        allowed_species=["Ti", "Fe", "Ni"],
        packing_fraction=0.65
    )
    
    # Save consolidated used seeds for next run
    parallel_gen.save_all_used_seeds(results)
    
    # Continue from previous generation (reuse seeds)
    parallel_gen_2 = ParallelBatchGenerator(
        generator_class=GeneratorRandomPacking,
        num_structures=300,
        num_processes=4,
        output_dir=Path("parallel_structures_2"),
        existing_used_seeds=Path("parallel_structures/used_seeds.txt"),  # Exclude previous seeds
        seeds_per_process=8_000
    )
    
    results_2 = parallel_gen_2.generate(
        atom_count=16,
        packing_fraction=0.62
    )
    
    What gets created:
    • process_00/, process_01/, ... subdirectories
    • Each contains: p00_POSCAR_0000, p00_POSCAR_0001, ...
    • Each contains: used_seeds_p00.txt, failed_seeds_p00.txt
    • Consolidated: used_seeds.txt (all allocated seeds for next run)
    
    Key improvements:
    • Transparent seed allocation (seeds_per_process parameter)
    • No seed overlaps between processes (guaranteed ranges)
    • Can reuse previous generations (existing_used_seeds parameter)
    • Single consolidated seed file for next run
"""
    print(readme_text)


def test_parallel_generation() -> Dict[str, Any]:
    """
    Test parallel generation with 10 structures across 3 processes.
    """
    from vsf.generate.generator_random import GeneratorRandomPacking
    from vsf.utils.io import flatten_dirs

    print("Starting parallel generation test...")
    print("Config: 10 structures, 3 processes, 9 atoms each")
    print("Seeds per process: 100 (small for testing)")
    print("Packing fraction: 0.55 (looser than default)")
    print("Output: test_structures/")
    print("-" * 50)

    # Set up parallel generator
    parallel_gen = ParallelBatchGenerator(
        generator_class=GeneratorRandomPacking,
        num_structures=10,
        num_processes=3,
        output_dir=Path("test_structures"),
        seeds_per_process=100,  # Small allocation for testing
    )

    # Run generation
    results = parallel_gen.generate(
        atom_count=9,
        packing_fraction=0.55,  # Looser packing for better success rate
        # All other parameters use defaults (all transition metals, etc.)
    )

    # Save consolidated seeds
    parallel_gen.save_all_used_seeds(results)

    print("\nGeneration Results:")
    print("=" * 50)
    pprint.pprint(results, indent=2)
    print("=" * 50)

    print("\nFlatten Dir structure: via `vsf.utils.io`")
    flatten_dirs(results["output_dirs"], Path("test_structures_flat"))

    # Quick summary
    print(f"\nSummary:")
    print(f"✓ Successfully generated: {results['total_success']}/10 structures")
    print(f"✗ Failed attempts: {results['total_failed']}")
    print(f"📁 Output directory: test_structures/")
    print(f"📄 Files: POSCAR_0000 through POSCAR_{results['total_success']-1:04d}")

    if results["total_success"] == 10:
        print("🎉 All structures generated successfully!")
    elif results["total_success"] > 0:
        print("⚠️  Partial success - some structures generated")
    else:
        print("❌ No structures generated - try looser packing parameters")

    return results


if __name__ == "__main__":
    print_readme()
    test_parallel_generation()
