import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from pymatgen.core import Structure
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Incar, Potcar
from pymatgen.io.vasp.sets import VaspInputSet

from .sets import MPRelaxSet2025_3_10, load_yaml_config

LOGGER = logging.getLogger(__name__)


def get_recommended_encut(potcar_path: str | Path, factor: float = 1.3) -> float:
    """
    Determines the recommended ENCUT value for a VASP calculation based on
    the maximum ENMAX value found in a POTCAR file, multiplied by a factor.

    Args:
        potcar_path: Path to the POTCAR file
        factor: Multiplication factor to apply to the maximum ENMAX (default=1.3)

    Returns:
        float: Recommended ENCUT value

    Examples:
        >>> encut = get_recommended_encut("POTCAR")
        >>> print(f"Recommended ENCUT: {encut} eV")
    """
    # Convert to Path object if string is provided
    potcar_path = Path(potcar_path)

    # Check if file exists
    if not potcar_path.exists():
        raise FileNotFoundError(f"POTCAR file not found at {potcar_path}")

    try:
        # Read the POTCAR file
        potcar = Potcar.from_file(potcar_path)

        # Extract the ENMAX values for each potential
        enmax_values = [pot.keywords["ENMAX"] for pot in potcar]

        # Find the maximum ENMAX value
        max_enmax = max(enmax_values)

        # Calculate the recommended ENCUT value
        recommended_encut = max_enmax * factor

        return recommended_encut
    except Exception as e:
        raise ValueError(f"Error reading POTCAR file: {e}")


def save_potcar_from_set(
    structure: Structure,
    potcar_name: str,
    potcar_dir: str | Path,
    input_set_source: Type[VaspInputSet] | str = MPRelaxSet2025_3_10,
) -> None:
    """
    Generate and save a POTCAR file for a given structure using the specified input set source.

    Args:
        structure: Pymatgen Structure object
        potcar_name: Output filename (without extension)
        potcar_dir: Directory to save the POTCAR file
        input_set_source: VaspInputSet class or YAML config name (default: MPRelaxSet2025_3_10)
    """

    # Handle input_set_source - could be a class or YAML string
    if isinstance(input_set_source, str):
        # Create dynamic VaspInputSet class from YAML config
        yaml_config = load_yaml_config(input_set_source)

        @dataclass
        class DynamicVaspInputSet(VaspInputSet):
            CONFIG = yaml_config

        input_set_class = DynamicVaspInputSet
        set_name = input_set_source  # YAML config name
        LOGGER.info(f"Using custom YAML config for POTCAR: {input_set_source}")
    else:
        input_set_class = input_set_source
        set_name = input_set_class.__name__  # Class name
        LOGGER.info(f"Using VaspInputSet class for POTCAR: {input_set_class.__name__}")

    # Create a VaspInputSet instance
    input_set = input_set_class(structure)

    # Save POTCAR file
    potcar_path = Path(potcar_dir, potcar_name)
    input_set.potcar.write_file(potcar_path)
    LOGGER.info(f"POTCAR saved to {potcar_path} using set: {set_name}")


def save_incar_from_set(
    structure: Structure,
    incar_name: str,
    incar_dir: str | Path,
    potcar_name: str | None = "POTCAR",
    encut_factor: float = 1.3,
    custom_incar_params: dict | None = None,
    input_set_source: Type[VaspInputSet] | str = MPRelaxSet2025_3_10,
    use_blank_incar: bool = True,
) -> None:
    """
    Generate and save an INCAR file for a given structure with an automatically
    adjusted ENCUT value if potcar_path is provided and optional custom parameters.

    Args:
        structure: Pymatgen Structure object
        incar_name: Output filename (without extension)
        incar_dir: Directory to save the INCAR file
        potcar_name: Optional POTCAR file name to determine ENCUT value
        encut_factor: Factor to multiply the maximum ENMAX by (default: 1.3)
        custom_incar_params: Dictionary of custom INCAR parameters to override defaults
        input_set_source: VaspInputSet class or YAML config name (default: MPRelaxSet2025_3_10)
        use_blank_incar: If True, starts with blank INCAR using only MAGMOM from input set.
                        If False, uses full parameter set from input_set_source (default: True)
    """

    # Handle input_set_source - could be a class or YAML string
    if isinstance(input_set_source, str):
        # Create dynamic VaspInputSet class from YAML config
        yaml_config = load_yaml_config(input_set_source)

        @dataclass
        class DynamicVaspInputSet(VaspInputSet):
            CONFIG = yaml_config

        input_set_class = DynamicVaspInputSet
        set_name = input_set_source  # YAML config name
        LOGGER.info(f"Using custom YAML config: {input_set_source}")
    else:
        input_set_class = input_set_source
        set_name = input_set_class.__name__  # Class name
        LOGGER.info(f"Using VaspInputSet class: {input_set_class.__name__}")

    # Create a VaspInputSet instance
    input_set = input_set_class(structure)

    # Get INCAR as dictionary for easier modification
    incar_dict = input_set.incar.as_dict()

    # If use_blank_incar is True, start with blank INCAR but keep MAGMOM
    if use_blank_incar:
        # Save the MAGMOM value if it exists
        magmom = incar_dict.get("MAGMOM", None)

        # Create a new dictionary with only the MAGMOM parameter
        if magmom is not None:
            incar_dict = {"MAGMOM": magmom}
            LOGGER.info("Starting with blank INCAR, keeping only MAGMOM from input set")
        else:
            incar_dict = {}
            LOGGER.info("Starting with blank INCAR, no MAGMOM found in input set")
    else:
        LOGGER.info("Using full parameter set from input set source")

    # Automatically adjust ENCUT if POTCAR path is provided
    if potcar_name is not None:
        recommended_encut = get_recommended_encut(
            Path(incar_dir, potcar_name), factor=encut_factor
        )
        # Override the ENCUT parameter in the dictionary
        incar_dict["ENCUT"] = recommended_encut
        LOGGER.info(f"ENCUT set to {recommended_encut:.2f} eV (factor: {encut_factor})")

    # Apply any custom INCAR parameters
    if custom_incar_params:
        incar_dict.update(custom_incar_params)
        LOGGER.info(f"Applied custom INCAR parameters: {custom_incar_params}")

    # Create new Incar object from the modified dictionary
    modified_incar = Incar.from_dict(incar_dict)

    # Save INCAR file
    incar_path = Path(incar_dir, incar_name)
    modified_incar.write_file(incar_path)
    LOGGER.info(f"INCAR saved to {incar_path} using set: {set_name}")


def save_kpoints_mueller(
    kpoints_path: Path,
    min_distance: float = 30.0,
    include_gamma: str = "auto",
    precalc_params: dict | None = None,
    save_precalc: bool = True,
) -> bool:
    """
    Generate and save a KPOINTS file for a given structure using kpoints_generator.

    Args:
        kpoints_path: Path to save the KPOINTS file
        min_distance: Minimum distance parameter for k-point generation (default: 35.0)
        include_gamma: Whether to include gamma point, options: TRUE, FALSE, AUTO
        precalc_params: Additional parameters for the PRECALC file (default: None)
        save_precalc: Whether to save the PRECALC file (default: True)

    Returns:
        Success status: bool
    """
    from .kpoints.generator import generate_kpoints

    # Create the output file path
    output_file = kpoints_path / "KPOINTS"

    # Prepare precalc parameters
    params = {} if precalc_params is None else precalc_params.copy()

    # Add include_gamma parameter if not already in precalc_params
    if "INCLUDEGAMMA" not in params:
        params["INCLUDEGAMMA"] = include_gamma.upper()

    # Generate k-points using kpoints_generator
    success, error_msg = generate_kpoints(
        mindistance=min_distance,
        kpoints_path=kpoints_path,
        precalc_params=params,
        output_file=str(output_file.name),
        save_precalc=save_precalc,
    )

    # Return just the success status - error details are already logged by generate_kpoints
    return success
