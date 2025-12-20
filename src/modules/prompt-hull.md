# Hull metrics

I have the following flat dir structure:

```
Cu3Pd_mp-580357_0.8
Cu3Pd_mp-580357_1
Cu3Pd_mp-580357_1.2
Cu3Pd_mp-672265_0.8
Cu3Pd_mp-672265_1
Cu3Pd_mp-672265_1.2
```

in these example there are 2 different structures with suffixes ranging always from 0.8 to 1.2 which indicates how much the poscar was stretched relative to 1.0
There is also a json file in each dir with the same name as dir + .json. Json contains parsed data. The idea is to go through all of the jsons and first of all plot all the curves for a given structure ox: volume coef, oy: potential_energy.

```json
{
  "potential_energy": {
    "results": {
      "VASP": {
        "value": -3.74203242
      },
      "MACE_MPA_0": {
        "value": -3.740002977160179
      },
      "ORBV3": {
        "value": -3.727995665431484
      }
    }
  }
}
```

This is the first step. Then we will og from there

# update plotting api

## API Migration Reference

### Old Pattern (deprecated):

```python
# eform_binary_heatmap.py
plot_heatmap_formation(structures, energy_source, save_path, figure_size=(10, 8),
                      dpi=None, vmin=None, vmax=None, cmap=None, annot=False)

plot_heatmap_delta(structures, energy_source, save_path, figure_size=(10, 8),
                  dpi=None, vmin=None, vmax=None, cmap=None, annot=False,
                  bar_title="Formation Energy Delta (eV/atom)")

# eform_distribution.py
plot_eform_distribution(structures, energy_source, energy_cutoff=None, bin_width=0.1,
                       num_bins=None, figsize=(8, 5), xlim=None, ylim=None,
                       dpi=None, save_path=None, color=None)

# eform_scatter.py
plot_energy_vs_scatter(structures, energy_source1, energy_source2, data_type="value",
                      gridsize=40, dpi=None, save_path=None, figsize=(8, 6),
                      xlim=None, ylim=None, x_range=None, y_range=None, dark=False)
```

### New Pattern (use config dataclasses):

```python
from vsf.core.plot.eform_binary_heatmap import HeatmapPlotConfig, plot_formation_energy_heatmap, plot_delta_heatmap
from vsf.core.plot.eform_distribution import DistributionPlotConfig, plot_formation_energy_distribution
from vsf.core.plot.eform_scatter import ScatterPlotConfig, plot_energy_comparison_scatter

# Binary heatmap: formation energy
config = HeatmapPlotConfig(figsize=(10, 8), vmin=None, vmax=None,
                           cmap=None, annot=False, dpi=None)
fig = plot_formation_energy_heatmap(structures, energy_source, config)

# Binary heatmap: delta
config = HeatmapPlotConfig(figsize=(10, 8), bar_title="Formation Energy Delta (eV/atom)",
                           vmin=None, vmax=None, cmap=None, annot=False, dpi=None)
fig = plot_delta_heatmap(structures, energy_source, reference_source="VASP", config=config)

# Distribution histogram
config = DistributionPlotConfig(figsize=(8, 5), xlim=None, ylim=None,
                                bin_width=0.1, num_bins=None, color=None,
                                dpi=None, energy_cutoff=None)
fig = plot_formation_energy_distribution(structures, energy_source, config, save_path)

# Scatter plot
config = ScatterPlotConfig(figsize=(8, 6), xlim=None, ylim=None, dpi=100,
                          show_diagonal=True, data_type="value",
                          x_range=None, y_range=None)
fig = plot_energy_comparison_scatter(structures, energy_source1, energy_source2,
                                     config, save_path)
```

## Migration Rules

1. **Group scattered parameters into config dataclasses** based on the new signatures
2. **Extract save_path to be a separate argument** (no longer embedded in method)
3. **Update imports** to use new function names and config classes
4. **Replace old method calls** on FormationEnergyFilter with direct function calls
5. **Config defaults match old defaults** — only specify parameters that differ from defaults

## Task

Refactor all plotting calls in the provided code to use the new config-based API.

- Keep logic identical
- Only change parameter passing and imports
- If code uses FormationEnergyFilter plotting methods, replace with direct function calls
- No new functionality needed

