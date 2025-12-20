# ============================================================================
# Example 1: Formation Energy Heatmap
# ============================================================================

from plot.eform_binary_heatmap import HeatmapPlotConfig, plot_formation_energy_heatmap

# Simple case: use all defaults
fig = plot_formation_energy_heatmap(structures, energy_source="MACE")

# With custom config: override figsize and annotation
config = HeatmapPlotConfig(figsize=(12, 10), annot=True, vmin=-0.5, vmax=0.5, dpi=150)
fig = plot_formation_energy_heatmap(structures, energy_source="SevenNet", config=config)


# ============================================================================
# Example 2: Formation Energy Distribution
# ============================================================================

from plot.eform_distribution import (
    DistributionPlotConfig,
    plot_formation_energy_distribution,
)

# Simple: defaults only
fig = plot_formation_energy_distribution(structures, energy_source="VASP")

# With cutoff for dual-color histogram and custom binning
config = DistributionPlotConfig(
    figsize=(10, 6),
    energy_cutoff=0.1,  # Splits histogram into stable/unstable
    bin_width=0.05,
    xlim=(-1, 3),
    dpi=150,
)
fig = plot_formation_energy_distribution(
    structures,
    energy_source="MACE",
    config=config,
    save_path="formation_energy_dist.png",
)


# ============================================================================
# Example 3: Energy Comparison Scatter
# ============================================================================

from plot.eform_scatter import ScatterPlotConfig, plot_energy_comparison_scatter

# Simple: VASP vs GNN
fig = plot_energy_comparison_scatter(structures, "VASP", "MACE")

# With range filtering and custom styling
config = ScatterPlotConfig(
    figsize=(10, 8),
    x_range=(-2, 5),  # Only plot VASP energies in this range
    y_range=(-2, 5),  # Only plot MACE energies in this range
    xlim=(-1, 4),  # Set axis display limits
    ylim=(-1, 4),
    show_diagonal=True,
    dpi=150,
)
fig = plot_energy_comparison_scatter(
    structures,
    energy_source1="VASP",
    energy_source2="SevenNet",
    config=config,
    save_path="scatter_comparison.png",
)
