import logging
from pathlib import Path

from vsf.core.record import StructureRecord
from vsf.energy.energy_source import EnergySource
from vsf.logging import setup_logging

LOGGER = logging.getLogger(__name__)
setup_logging(log_file="x-test.log", console_level=logging.INFO)

base_dir = Path("e-1-gnncp.test")

json_paths = list(base_dir.rglob("**_full.json"))

for p in json_paths:
    record = StructureRecord.load_json(structure_dir=p.parent)
    assert record is not None

    # Check if delete actually worked in memory
    record.potential_energy.delete(EnergySource.SEVENNET)
    deleted = record.formation_energy.delete(EnergySource.SEVENNET)
    print(f"Deleted: {deleted}")

    record.save_json(overwrite=True)
