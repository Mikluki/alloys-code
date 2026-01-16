import logging
from pathlib import Path

from vsf.liquid.diagnostic.rdf import batch_rdf_analysis
from vsf.logging import setup_logging

LOGGER = setup_logging(log_file="x-31-check-rdf.log", console_level=logging.DEBUG)

base_dir = Path("x-all300k2-decorr-poscar")
element = ""

# GLOB Poscar paths
poscar_paths = list(base_dir.rglob("**/POSCAR"))
poscar_paths = [p for p in poscar_paths if element in p.parent.name]
[LOGGER.info(f"{p}") for p in poscar_paths]

if not poscar_paths:
    LOGGER.warning(f"No POSCAR files found for element {element} in {base_dir}")

else:
    batch_rdf_analysis(
        poscar_paths,
        output_dir=base_dir,
        bin_width=0.1,
        save_plot=False,
        max_workers=8,
    )
