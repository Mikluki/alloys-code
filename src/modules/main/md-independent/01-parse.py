import json
import logging
from pathlib import Path

from vsf.core.md_independent.outcar_parser import parse_outcar
from vsf.logging import setup_logging

(log_dir := Path("logs")).mkdir(exist_ok=True)
LOGGER = setup_logging(
    log_file=log_dir / "x-parse-outcar.log",
    console_level=logging.DEBUG,
    file_level=logging.DEBUG,
)

run_dir = Path("data/AlCuNi_L1915_1400")
outcar = run_dir / "OUTCAR"
poscar = run_dir / "POSCAR"

steps = parse_outcar(outcar_path=outcar, poscar_path=poscar)

dic = steps[0].to_dict()

with open(f"{run_dir.name}.json", "w") as f:
    json.dump(dic, f, indent=2)
