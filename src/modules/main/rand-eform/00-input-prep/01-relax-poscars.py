import json
import logging
from pathlib import Path

from vsf.calculators import Mace_mpa_0
from vsf.logging import setup_logging
from vsf.transform.poscars_relax import (
    ChainedRelaxation,
)

LOGGER = logging.getLogger(__name__)
setup_logging(log_file="relax.log")

base_dir = Path("rand-init")
key = "POSCAR"
base_output_dir = "relaxed"
FMAX = 0.02

# Initialize calculator
calc = Mace_mpa_0()
calc.initialize(default_dtype="float64")

key_file = "POSCAR"
key_dir = "rand"

poscar_paths = [
    p
    for p in base_dir.rglob(key_file)
    if p.parent.parent == base_dir and key_dir in p.parent.name
]

# Chained relaxation workflow - isolated stages
chain = ChainedRelaxation(poscar_paths, calc, base_output_dir=base_output_dir)

# Define relaxation stages
chain.add_stage("01_csym", {"constant_symmetry": True, "fmax": FMAX})
chain.add_stage("02_full", {"fmax": FMAX})

# Run all stages
all_results = chain.run_all_stages()

with open("all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
