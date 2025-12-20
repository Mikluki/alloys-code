import logging
from pathlib import Path

from vsf.dataset.leaderboard import generate_leaderboard
from vsf.logging import setup_logging

(log_dir := Path("xds-metrics/logs")).mkdir(exist_ok=True)
LOGGER = setup_logging(
    log_file=log_dir / f"x-eform.log",
    console_level=logging.INFO,
    file_level=logging.INFO,
)

metrics_file = Path("xds-metrics/aggregated_metrics_T001.json")

# Formation Energy Leaderboard
generate_leaderboard(
    metrics_file=metrics_file,
    output_file=Path("xds-metrics/x-leaderboard_formation_energy.md"),
    property_name="formation_energy",
    show_all_metrics=True,
)

# Stress/Pressure Leaderboard
generate_leaderboard(
    metrics_file=metrics_file,
    output_file=Path("xds-metrics/x-leaderboard_stress_pressure.md"),
    property_name="stress_pressure",
    show_all_metrics=True,
)
