import logging
import subprocess
from pathlib import Path
from typing import List

from pymatgen.core import Structure

LOGGER = logging.getLogger(__name__)


def flatten_dirs(
    source_dirs: List[Path],
    target_dir: Path,
) -> int:
    """
    Flatten all files from source directories into target directory using bash.

    Copies all files from source directories to target directory in one level.
    Fast and generic - works with any file types.

    Parameters
    ----------
    source_dirs : List[Path]
        List of source directories to flatten
    target_dir : Path
        Target directory to copy all files into

    Returns
    -------
    int
        Number of files in target directory after flattening
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(exist_ok=True, parents=True)

    # Filter existing directories and build source patterns
    source_patterns = []
    for d in source_dirs:
        if d.exists() and d.is_dir():
            source_patterns.append(str(d / "*"))

    if not source_patterns:
        LOGGER.warning("No valid source directories found")
        return 0

    # Use bash to copy all files efficiently
    bash_cmd = f"cp {' '.join(source_patterns)} {target_dir}/"

    try:
        subprocess.run(
            ["bash", "-c", bash_cmd], check=True, capture_output=True, text=True
        )
        LOGGER.info(
            f"Flattened files from {len(source_dirs)} directories to {target_dir}"
        )
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Failed to flatten directories: {e.stderr}")
        raise

    # Count files in target directory
    file_count = len([f for f in target_dir.iterdir() if f.is_file()])
    return file_count


def load_structure_from_poscar(poscar_path: str | Path) -> Structure:
    """Load pymatgen Structure from POSCAR file."""
    poscar_path = Path(poscar_path)
    if not poscar_path.exists():
        raise FileNotFoundError(f"POSCAR file not found: {poscar_path}")

    with open(poscar_path, "r", encoding="utf-8") as f:
        poscar_content = f.read()

    return Structure.from_str(poscar_content, fmt="poscar")
