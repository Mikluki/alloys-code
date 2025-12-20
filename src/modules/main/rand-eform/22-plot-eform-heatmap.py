import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from vsf.core.analysis import FormationEnergyFilter, ResultsAggregator
from vsf.energy.energy_source import EnergySource
from vsf.logging import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(log_file="x-eform-plot-destr.log", console_level=logging.INFO)

energy_cutoff = 0.010  # in eV
energy_sources = [
    EnergySource.VASP.value,
    EnergySource.MACE.value,
    EnergySource.MACE_MPA_0.value,
    EnergySource.ESEN_30M_OAM.value,
    EnergySource.ORBV3.value,
    EnergySource.SEVENNET.value,
]

# Multi-directory aggregation and filtering
directories = [
    Path("e-01"),
    Path("e-02"),
    Path("e-03"),
    Path("e-04"),
    Path("e-05"),
    Path("e-06"),
    Path("e-07"),
    Path("e-08"),
    Path("e-09"),
    Path("e-10"),
    Path("e-11"),
]

# Aggregate results from all directories
aggregator = ResultsAggregator(directories)
all_results = aggregator.load_all_results()

# Save massive json result
# all_results = json.load(open("all_results.json"))
json.dump(all_results, open("all_results.json", "w"), indent=2)

# Filter stable structures
filter_tool = FormationEnergyFilter()

for energy_source in energy_sources:

    figure_save_path = Path(f"destr_efrom.{energy_source}.png")
    filter_tool.plot_eform_distribution(
        structures=all_results,
        energy_source=energy_source,
        energy_cutoff=energy_cutoff,
        bin_width=0.1,
        num_bins=100,
        xlim=(-2, 3),
        save_path=figure_save_path,
    )
