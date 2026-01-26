import logging
from pathlib import Path

from vsf.core.hull.metrics import (
    plot_eos_benchmark,
    plot_eos_benchmark_by_method,
    plot_hull_metrics,
    plot_pressure_metrics,
)
from vsf.logging import setup_logging

(log_dir := Path("logs")).mkdir(exist_ok=True)
LOGGER = setup_logging(
    log_file=log_dir / "x-hull-plt.log",
    console_level=logging.DEBUG,
    file_level=logging.DEBUG,
)

path = Path("outcar-hull")
output = Path("pics")
output_pressure = Path("pics/pressure")
output_eos_by_struct = Path("pics/eos-by-struct")
output_eos_by_method = Path("pics/eos-by-method")

plot_hull_metrics(base_path=path, output_dir=output)

# plot_pressure_metrics(base_path=path, output_dir=output_dir_pressure)

plot_eos_benchmark(base_path=path, output_dir=output_eos_by_struct)

plot_eos_benchmark_by_method(base_path=path, output_dir=output_eos_by_method)
