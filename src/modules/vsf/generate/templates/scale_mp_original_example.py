from pathlib import Path

from vsf.generation.scale_poscars import scale_poscar_lattice

# Get all POSCAR files in the mp_original directory using glob
mp_dir = Path("mp_original")
poscar_files = list(mp_dir.glob("*.poscar"))

# Create output directory as a Path
output_dir = "scaled"

# Call the function with string paths that it can handle
scale_poscar_lattice(
    poscar_files,  # Using string paths instead of Path objects
    scale_factor=2,
    output_dir=str(output_dir),  # Convert Path to string
    prefix="2x_",
)
