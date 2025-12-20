import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

LOGGER = logging.getLogger(__name__)


def get_parent_stem(path: Path) -> str:
    """
    For Path returns string of parent/stem
    """
    return f"{path.parent.stem}/{path.name}"


@dataclass
class EntryPaths:
    """Manages all file paths for an Entry"""

    entry_dir: Path
    poscar_master: Path
    poscar: Path
    incar: Path
    potcar: Path
    kpoints: Path
    json: Path
    outcar: Path

    @classmethod
    def from_entry_dir(cls, entry_dir: Path, name: str):
        """Create paths from base directory and entry name"""
        # entry_dir_format = f"{name}_"
        # if str(entry_dir.stem) != entry_dir_format:
        #     raise ValueError(
        #         f"{get_parent_stem(entry_dir)} name does not follow {entry_dir_format}. Entry_dir will be overwritten"
        #     )

        return cls(
            entry_dir=entry_dir,
            poscar_master=entry_dir / f"{name}.poscar",
            potcar=entry_dir / "POTCAR",
            incar=entry_dir / "INCAR",
            kpoints=entry_dir / "KPOINTS",
            poscar=entry_dir / "POSCAR",
            json=entry_dir / f"{name}.json",
            outcar=entry_dir / "OUTCAR",
        )

    def initialize(self) -> None:
        """Create the data directory if it doesn't exist"""
        if not self.entry_dir.exists():  # Check if the directory does not exist
            ENTRY_LOGGER.warning(f"mkdir {get_parent_stem(self.entry_dir)}")
            self.entry_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict:
        """Convert paths to dictionary."""
        return {
            "entry_dir": get_parent_stem(self.entry_dir),
            "poscar_master": get_parent_stem(self.poscar_master),
            "json": get_parent_stem(self.json),
            "poscar": get_parent_stem(self.poscar),
            "incar": get_parent_stem(self.incar),
            "potcar": get_parent_stem(self.potcar),
            "kpoints": get_parent_stem(self.kpoints),
        }
