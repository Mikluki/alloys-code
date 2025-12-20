import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from vsf.core.analysis import FormationEnergyFilter, ResultsAggregator
from vsf.energy.energy_source import EnergySource
from vsf.logging import setup_logging

LOGGER = setup_logging(log_file="x-eform-plot-destr.log", console_level=logging.INFO)

energy_cutoff = 0.010  # in eV

vasp_source = EnergySource.VASP.value
energy_sources = [
    EnergySource.VASP.value,
    # EnergySource.MACE.value,
    # EnergySource.MACE_MPA_0.value,
    # EnergySource.ESEN_30M_OAM.value,
    # EnergySource.ORBV3.value,
    # EnergySource.SEVENNET.value,
]

# Multi-directory aggregation and filtering
directories = [
    Path("x-cp"),
]

# Aggregate results from all directories
aggregator = ResultsAggregator(directories)
all_results = aggregator.load_all_results()

# Save massive json result
# all_results = json.load(open("all_results.json"))
json.dump(all_results, open("all_results.json", "w"), indent=2)

# Filter stable structures
filter_tool = FormationEnergyFilter()
vasp_paths = filter_tool.filter_by_threshold(
    all_results, energy_source=vasp_source, energy_cutoff=energy_cutoff
)

for energy_source in energy_sources:

    source_paths = filter_tool.filter_by_threshold(
        all_results, energy_source=energy_source, energy_cutoff=energy_cutoff
    )

    # Count how many vasp_paths are in source_paths
    common_count = len(set(vasp_paths) & set(source_paths))
    LOGGER.info(
        f"{energy_source}: {common_count}/{len(vasp_paths)} correctly classified structures"
    )

    missed_by_gnn_count = len(set(vasp_paths) - set(source_paths))
    LOGGER.info(
        f"{energy_source}: {missed_by_gnn_count} stable structures misclassified as unstable"
    )

    # GNN stable that VASP says unstable (false positives)
    gnn_false_positive = set(source_paths) - set(vasp_paths)
    LOGGER.info(
        f"{energy_source}: {len(gnn_false_positive)} unstable structures misclassified as stable"
    )

    figure_save_path = Path(f"destr_efrom.{energy_source}.svg")
    filter_tool.plot_eform_distribution(
        structures=all_results,
        energy_source=energy_source,
        energy_cutoff=energy_cutoff,
        bin_width=0.1,
        num_bins=60,
        figsize=(6.0, 3.0),
        # figsize=(5.0, 4.0),
        # ylim=(0, 500),
        xlim=(0, 0.4),
        save_path=figure_save_path,
    )
