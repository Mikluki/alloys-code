import logging
from datetime import datetime
from pathlib import Path

from vsf.dataset.metrics import process_datasets
from vsf.logging import setup_logging

timestamp = datetime.now().strftime("%m-%dT%H:%M:%S")
(log_dir := Path("xds-metrics/logs")).mkdir(exist_ok=True)
LOGGER = setup_logging(
    log_file=log_dir / f"x-eform_{timestamp}.log",
    console_level=logging.INFO,
    file_level=logging.INFO,
)

base_dirs = [
    # "w-outcar-pure-gnncp",
    "x-cat-1-gnncp",
    "x-cat-2-gnncp",
    "x-hull-gnncp",
    "x-liquid-k1-gnncp",
    "x-liquid-k2-gnncp",
    "x-rand-01-gnncp",
    "x-rand-02-gnncp",
]

base_dirs = [Path(f"xds-metrics/xout/{d}") for d in base_dirs]
output_directory = Path("xds-metrics")

process_datasets(
    base_directories=base_dirs,
    output_dir=output_directory,
    stability_threshold=0.010,
)
