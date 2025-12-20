import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from vsf.core.analysis import FormationEnergyFilter, ResultsAggregator
from vsf.core.plot.eform_binary_heatmap import plot_delta_heatmap
from vsf.energy.energy_source import EnergySource
from vsf.logging import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(log_file="x-eform-plot-heatmap.log", console_level=logging.INFO)

energy_cutoff = 0.010  # in eV
# energy_source = EnergySource.MACE.value
# energy_source = EnergySource.MACE_MPA_0.value
# energy_source = EnergySource.ESEN_30M_OAM.value
# energy_source = EnergySource.ORBV3.value
energy_source = EnergySource.SEVENNET.value

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

full_cmap = matplotlib.colormaps["magma_r"]
clamped_cmap = ListedColormap(full_cmap(np.linspace(0, 0.65, 256)))

plot_path = Path(f"heatmap_delta_ref.vasp.vs.{energy_source}.svg")
filter_tool.plot_heatmap_delta(
    structures=all_results,
    energy_source=energy_source,
    save_path=plot_path,
    figure_size=(7, 6),
    vmin=None,
    vmax=0.5,
    cmap=None,
    annot=False,
)

plt.show()
