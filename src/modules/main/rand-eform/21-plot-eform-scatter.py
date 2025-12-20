import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from vsf.core.analysis import FormationEnergyFilter, ResultsAggregator
from vsf.energy.energy_source import EnergySource
from vsf.logging import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(log_file="x-eform-plot-scatter.log", console_level=logging.DEBUG)

energy_source1 = EnergySource.VASP.value
e_sources2 = [
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


for energy_source2 in e_sources2:
    plot_path = Path(f"scatter.{energy_source1}.vs.{energy_source2}.svg")
    filter_tool.plot_energy_vs_scatter(
        structures=all_results,
        energy_source1=energy_source1,
        energy_source2=energy_source2,
        data_type="value",
        save_path=plot_path,
        gridsize=150,
        figsize=(5, 4),
        dark=False,
        xlim=(-2, 3),
        ylim=(-2, 3),
        use_scatter=True,
        show_n_structures=False,
    )

    # plt.show()
