import logging
from pathlib import Path

from vsf.core.md_stability.plot import plot_force_collapse
from vsf.logging import setup_logging

# Logging
(log_dir := Path("logs")).mkdir(exist_ok=True)
LOGGER = setup_logging(
    log_file=log_dir / "x-gnn-md-post-plot.log",
    console_level=logging.DEBUG,
    file_level=logging.DEBUG,
)

path = Path("results-MACE_MPA_0-12-01T17:00:38")
traj_json_path = path / "seed-0/trajectory.json"

plot_force_collapse(
    traj_json_path,
    output_path=Path("collapse_plot.png"),
)
