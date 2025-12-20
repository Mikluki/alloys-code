import logging
from pathlib import Path

from vsf.energy.energy_source import EnergySource
from vsf.energy.workflow_vasp import VaspEnergyWorkflow
from vsf.logging import setup_logging
from vsf.properties.workflow_vasp_data import VaspDataWorkflow

(log_dir := Path("logs")).mkdir(exist_ok=True)
LOGGER = setup_logging(
    log_file=log_dir / "x-outcar-extract-ewe-stress.log",
    console_level=logging.INFO,
    file_level=logging.INFO,
)

target_dirs = [
    # Path("outcar-cat-1"),
    # Path("outcar-cat-2"),
    Path("00-outcar-pure"),
]
hull_dir = Path("00-outcar-pure")

from_json = True
force_recalculate = True

for target_dir in target_dirs:
    LOGGER.info(f"Processing DIR: {target_dir.parent}/{target_dir.name}")
    vasp_energy_workflow = VaspEnergyWorkflow()

    # # Run workflow
    # results = vasp_energy_workflow.run_workflow(
    #     target_dir=target_dir,
    #     hull_dir=hull_dir,
    #     energy_source=EnergySource.VASP,
    #     force_recalculate=force_recalculate,
    #     from_json=from_json,
    # )

    vasp_data_workflow = VaspDataWorkflow()
    vasp_data_workflow.extract_target_structure_stresses(
        target_dir=target_dir,
        force_recalculate=force_recalculate,
        from_json=from_json,
    )
