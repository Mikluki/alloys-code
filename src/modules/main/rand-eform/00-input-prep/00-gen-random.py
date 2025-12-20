import logging
from datetime import datetime
from pathlib import Path

from vsf.generate.generator_parallel import ParallelBatchGenerator
from vsf.generate.generator_random import GeneratorRandomPacking
from vsf.logging import setup_logging
from vsf.transform.poscar_organizer import PoscarOrganizer

LOGGER = logging.getLogger(__name__)
setup_logging(log_file="gen.log")

# Your original configuration
base_dir = Path("init")
used_seeds = Path("all_seeds.txt")
num_structures = 3000
num_processes = 8
seeds_per_process = int(num_structures / num_processes) * 2  # two seeds per structure


# First generation
LOGGER.info("=== FIRST GENERATION ===")

parallel_gen = ParallelBatchGenerator(
    generator_class=GeneratorRandomPacking,
    num_structures=num_structures,
    num_processes=num_processes,
    output_dir=base_dir,
    existing_used_seeds=used_seeds,
    seeds_per_process=seeds_per_process,
)

results = parallel_gen.generate(
    atom_count=16,
    packing_fraction=0.62,
    safety_factor=1.0,
    max_attempts_per_atom=1_000_000,
)

LOGGER.info(
    f"First generation - Success: {results['total_success']}, Failed: {results['total_failed']}"
)

# Save all seeds for next run
timestamp = datetime.now().strftime("%yy%mm%dd_%H%M%S")
consolidated_seeds_file = base_dir / f"used_seeds_{timestamp}.txt"
parallel_gen.save_all_used_seeds(results, consolidated_seeds_file)


key = "POSCAR"
poscar_paths = [p for p in base_dir.rglob("*") if key in p.name]

# Organize input paths to dedicated dirs
starting_id = 2002000
organizer = PoscarOrganizer.from_starting_id(starting_id)
organized_dirs = organizer.organize_poscar_list(poscar_paths, base_dir, "rand")
