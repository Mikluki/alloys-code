import json
import logging
from pathlib import Path

from vsf.core.analysis import FormationEnergyFilter, ResultsAggregator
from vsf.energy.energy_source import EnergySource
from vsf.logging import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(log_file="x-filter.log", console_level=logging.DEBUG)


energy_sources = [EnergySource.VASP.value]
energy_cutoff = 0.010  # eV

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
# all_results = json.load(open("all_results.json"))
json.dump(all_results, open("all_results.json", "w"), indent=2)

# Filter stable structures
filter_tool = FormationEnergyFilter()


for e_source in energy_sources:

    stable_paths = filter_tool.filter_by_threshold(
        all_results, energy_cutoff, energy_source=e_source
    )

    stable_output_path = Path(f"e-stable-{e_source}.txt")
    filter_tool.write_poscar_paths(paths=stable_paths, output_file=stable_output_path)
