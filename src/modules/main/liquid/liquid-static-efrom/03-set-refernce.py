import logging
from pathlib import Path

from vsf.energy.energy_source import EnergySource
from vsf.energy.workflow_vasp import VaspEnergyWorkflow
from vsf.logging import setup_logging

(log_dir := Path("logs")).mkdir(exist_ok=True)
LOGGER = setup_logging(
    log_file=log_dir / "x-eform.log",
    console_level=logging.INFO,
    file_level=logging.INFO,
)

target_dirs = [
    Path("x-all300k-decorr-poscar"),
]

for td in target_dirs:
    vasp_workflow = VaspEnergyWorkflow()

    # Update reference sources
    updated_count = vasp_workflow.set_reference_sources(
        target_dir=td,
        potential_energy_source=EnergySource.VASP,
        formation_energy_source=EnergySource.VASP,
        overwrite=True,
    )
