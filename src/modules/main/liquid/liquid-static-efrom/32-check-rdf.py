import logging
from pathlib import Path

from vsf.energy.energy_source import EnergySource
from vsf.liquid.diagnostic.rdf import batch_rdf_analysis
from vsf.logging import setup_logging

LOGGER = setup_logging(log_file="x-31-check-rdf.log", console_level=logging.DEBUG)

energy_source1 = EnergySource.VASP.value
e_sources2 = [
    # EnergySource.MACE.value,
    EnergySource.MACE_MPA_0.value,
    # EnergySource.ESEN_30M_OAM.value,
    # EnergySource.ORBV3.value,
    # EnergySource.SEVENNET.value,
]


base_dir = Path("x-all300k-decorr-poscar")

sus_dirs = ["decorr_Ni_3001890"]

sus_paths = [Path(base_dir, d, "POSCAR") for d in sus_dirs]
element = ""

[LOGGER.info(f"{p}") for p in sus_paths]
if not sus_paths:
    LOGGER.warning(f"No POSCAR files found for element {element} in {base_dir}")

else:
    batch_rdf_analysis(
        sus_paths,
        output_dir=base_dir,
        bin_width=0.1,
        save_plot=True,
        max_workers=8,
    )
