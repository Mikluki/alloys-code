import logging
from pathlib import Path

from vsf.logging import setup_logging
from vsf.transform.poscar_organizer import PoscarOrganizer

LOGGER = logging.getLogger(__name__)
setup_logging(log_file="org.log")

# Your original configuration

base_dir = Path("init-e891011")

key = "POSCAR"
poscar_paths = [p for p in base_dir.rglob("*") if key in p.name]

# Organize input paths to dedicated dirs
starting_id = 2002000
organizer = PoscarOrganizer.from_starting_id(starting_id)
organized_dirs = organizer.organize_poscar_list(poscar_paths, base_dir, "rand")
