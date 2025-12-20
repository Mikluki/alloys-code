import json
import logging
from pathlib import Path

from vsf.core.analysis import FormationEnergyFilter, ResultsAggregator
from vsf.logging import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(log_file="x-eform-filter.log", console_level=logging.INFO)

energy_cutoff = 0.010  # in eV

# Multi-directory aggregation and filtering
directories = [
    Path("e-1"),
    Path("e-2"),
    Path("e-3"),
    Path("e-4"),
]

# Aggregate results from all directories
aggregator = ResultsAggregator(directories)
all_results = aggregator.load_all_results()

# Save massive json result
json.dump(all_results, open("all_results.json", "w"), indent=2)

# Filter stable structures (< 0.025 eV/atom)
filter_tool = FormationEnergyFilter()
stable_paths = filter_tool.filter_by_threshold(
    all_results,
    energy_cutoff=energy_cutoff,
)

# Write paths for bash processing
filter_tool.write_poscar_paths(stable_paths, Path("stable_structures.txt"))

# Get statistics
stats = filter_tool.get_formation_energy_stats(
    all_results,
    energy_cutoff,
    "VASP",
)
print(f"Formation energy statistics: {stats}")

json.dump(all_results, open("filter_stats.json", "w"), indent=2)
