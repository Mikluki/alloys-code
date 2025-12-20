import logging
from pathlib import Path

from vsf.logging import setup_logging
from vsf.vasp.input.files import VaspInputFiles

LOGGER = setup_logging(log_file=f"x-input.log", console_level=logging.INFO)


base_dir = Path("x-all300k2-decorr-poscar")

kpoints_include_gamma = "auto"
kpoints_min_distance = 28
incar_dic = {
    "ALGO": "Normal",
    "EDIFF": 1.0e-05,
    "GGA_COMPAT": False,
    "IBRION": -1,
    "ISMEAR": 1,
    "ISPIN": 2,
    "ISIF": 2,
    "LASPH": True,
    "LCHARG": False,
    "LMAXMIX": 6,
    "LREAL": "AUTO",
    "LWAVE": False,
    "NELM": 60,
    "NELMIN": 4,
    "NCORE": 4,
    "KPAR": 4,
    "PREC": "Accurate",
    "SIGMA": 0.1,
    "NSW": 0,
}

# GLOB Poscar paths
poscar_paths = list(base_dir.rglob("**/POSCAR"))

# Main batch processing loop
failed_dirs = []

for p in poscar_paths:
    print(p)
    vasp_inputs = VaspInputFiles(p)
    vasp_inputs.save_potcar()

    # Generate INCAR with your custom params
    vasp_inputs.save_incar(
        custom_incar_params=incar_dic,
        rewrite=False,
    )

    # Generate KPOINTS with defaults
    success = vasp_inputs.save_kpoints(
        min_distance=kpoints_min_distance,
        include_gamma=kpoints_include_gamma,
    )

    if not success:
        failed_dirs.append(str(p))

# Log failures at the end
if failed_dirs:
    LOGGER.info(f"Failed to generate KPOINTS for {len(failed_dirs)} directories")
    for d in failed_dirs:
        LOGGER.info(f"Failed: {d}")
else:
    LOGGER.info("All KPOINTS generation succeeded")
