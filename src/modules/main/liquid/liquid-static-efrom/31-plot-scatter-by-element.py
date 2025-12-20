import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from vsf.core.analysis import FormationEnergyFilter, ResultsAggregator
from vsf.core.plot.eform_scatter import add_stats_box, plot_energy_vs_scatter
from vsf.energy.energy_source import EnergySource
from vsf.logging import setup_logging

LOGGER = setup_logging(log_file="x-eform-plot-scatter.log", console_level=logging.DEBUG)

energy_source1 = EnergySource.VASP.value
e_sources2 = [
    # EnergySource.MACE.value,
    EnergySource.MACE_MPA_0.value,
    # EnergySource.ESEN_30M_OAM.value,
    # EnergySource.ORBV3.value,
    # EnergySource.SEVENNET.value,
]

# Multi-directory aggregation and filtering
directories = [
    Path("x-liquid-k1-gnncp"),
    Path("x-liquid-k2-gnncp"),
]

# Aggregate results from all directories
aggregator = ResultsAggregator(directories)
all_results = aggregator.load_all_results()

# Filter stable structures
filter_tool = FormationEnergyFilter()

# Per-element analysis
elements = ["Al", "Au", "Cu", "Na", "Ni"]

for energy_source2 in e_sources2:
    # Create one figure for all elements
    fig = None
    ax = None
    all_x_data = []
    all_y_data = []

    print(f"\n{'='*60}")
    print(f"Energy source comparison: {energy_source1} vs {energy_source2}")
    print(f"{'='*60}")

    for element in elements:
        elem_data = [r for r in all_results if set(r["atoms"].keys()) == {element}]

        print(f"\n--- {element} ({len(elem_data)} structures) ---")

        for d in elem_data:
            eform_vasp = float(d["formation_energy"]["results"]["VASP"]["value"])
            if eform_vasp > 0.5:
                LOGGER.info(f"High Eform: {eform_vasp} for {d['name']}")

        # Plot on same axes (colors auto-cycle)
        fig, ax, x_data, y_data = plot_energy_vs_scatter(
            structures=elem_data,
            energy_source1=energy_source1,
            energy_source2=energy_source2,
            data_type="value",
            figsize=(8, 6),
            fig=fig,
            ax=ax,
            label=element,
            show_diagonal=(element == elements[0]),  # Show diagonal only once
            # x_range=(-0.5, 0.5),
            # y_range=(-0.5, 0.5),
            # x_range=(-2, 6),
            # y_range=(-2, 6),
            # xlim=(-2, 6),
            # ylim=(-2, 6),
        )

        # Calculate and log per-element statistics
        n_points = len(x_data)
        mae = np.mean(np.abs(y_data - x_data))
        r_squared = r2_score(x_data, y_data)

        print(f"  N points: {n_points:,}")
        print(f"  MAE:      {mae:.3f} eV/atom")
        print(f"  R²:       {r_squared:.3f}")

        all_x_data.append(x_data)
        all_y_data.append(y_data)

    # Calculate and print combined statistics
    x_combined = np.concatenate(all_x_data)
    y_combined = np.concatenate(all_y_data)
    combined_n = len(x_combined)
    combined_mae = np.mean(np.abs(y_combined - x_combined))
    combined_r2 = r2_score(x_combined, y_combined)

    print(f"\n--- COMBINED (all elements) ---")
    print(f"  N points: {combined_n:,}")
    print(f"  MAE:      {combined_mae:.3f} eV/atom")
    print(f"  R²:       {combined_r2:.3f}")

    # Add combined statistics box to plot
    assert ax is not None and fig is not None
    add_stats_box(ax, all_x_data, all_y_data, show_n_points=True)

    plt.tight_layout()

    # Save figure
    plot_path = Path(
        "pics", f"x-scatter.{energy_source1}.vs.{energy_source2}.all-elements.png"
    )
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
