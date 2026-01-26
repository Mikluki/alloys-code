import json
import logging
from pathlib import Path

from vsf.core.md_stability.outcar_parser import parse_outcar

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

run_dir = Path("data/AlCuNi_L1915_1400")
outcar = run_dir / "OUTCAR"
poscar = run_dir / "POSCAR"

steps = parse_outcar(outcar_path=outcar, poscar_path=poscar)

dic = steps[0].to_dict()

with open(f"{run_dir.name}.json", "w") as f:
    json.dump(dic, f, indent=2)
