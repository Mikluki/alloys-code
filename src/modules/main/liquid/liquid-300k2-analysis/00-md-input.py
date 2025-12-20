#!/usr/bin/env python3
"""
Generate MD INCAR parameters for liquid metal simulations.
Works with VaspInputFiles workflow - creates multiple runs with different random seeds.
"""

import logging
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp import Kpoints

LOGGER = logging.getLogger(__name__)

POTIM = 1.0
SAMPLING_INTERVAL = 30
T_DELTA = 300

# Element-specific parameters
melt_temp_Al = 933
melt_temp_Na = 370
melt_temp_Au = 1337
melt_temp_Cu = 1358
melt_temp_Ni = 1728

ELEMENT_DATA = {
    "Al": {
        "melting_point": melt_temp_Al,
        "target_temp": melt_temp_Al + T_DELTA,
        "init_temp": melt_temp_Al + T_DELTA + 200,
    },
    "Na": {
        "melting_point": melt_temp_Na,
        "target_temp": melt_temp_Na + T_DELTA,
        "init_temp": melt_temp_Na + T_DELTA + 200,
    },
    "Au": {
        "melting_point": melt_temp_Au,
        "target_temp": melt_temp_Au + T_DELTA,
        "init_temp": melt_temp_Au + T_DELTA + 200,
    },
    "Cu": {
        "melting_point": melt_temp_Cu,
        "target_temp": melt_temp_Cu + T_DELTA,
        "init_temp": melt_temp_Cu + T_DELTA + 200,
    },
    "Ni": {
        "melting_point": melt_temp_Ni,
        "target_temp": melt_temp_Ni + T_DELTA,
        "init_temp": melt_temp_Ni + T_DELTA + 200,
    },
}

# Base static INCAR (your template)
STATIC_INCAR_BASE = {
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


def create_gamma_kpoints(poscar_path):
    """
    Create KPOINTS file with single gamma point in same directory as POSCAR.

    Args:
        poscar_path: Path to POSCAR file

    Returns:
        Path: Path to created KPOINTS file
    """
    poscar_path = Path(poscar_path)

    # Create gamma-point only KPOINTS
    kpoints = Kpoints.gamma_automatic(kpts=(1, 1, 1), shift=(0, 0, 0))

    # Set comment
    kpoints.comment = "Gamma"

    # Save in same directory as POSCAR
    kpoints_path = poscar_path.parent / "KPOINTS"
    kpoints.write_file(kpoints_path)

    return kpoints_path


def detect_element_from_poscar(poscar_path):
    """
    Detect primary element from POSCAR file.

    Args:
        poscar_path: Path to POSCAR file

    Returns:
        str: Element symbol (Al, Na, Au, Cu, Ni)
    """
    structure = Structure.from_file(poscar_path)
    composition = structure.composition

    # Get the most abundant element
    primary_element = composition.get_el_amt_dict()
    element = max(primary_element, key=primary_element.get)  # pyright: ignore

    return str(element)


def create_md_incar_params(
    element,
    random_seed=1,
    equilibration_steps=1500,
    sampling_configs=100,
    sampling_interval=SAMPLING_INTERVAL,
):
    """
    Create MD INCAR parameters dictionary for specific element.

    Args:
        element: Element symbol (Al, Na, Au, Cu, Ni)
        random_seed: Random seed for independent trajectories
        equilibration_steps: Steps for equilibration before sampling
        sampling_configs: Number of configurations to collect
        sampling_interval: Steps between sampled configurations

    Returns:
        dict: INCAR parameters for MD simulation
        dict: Simulation info
    """

    if element not in ELEMENT_DATA:
        raise ValueError(
            f"Element {element} not supported. Use: {list(ELEMENT_DATA.keys())}"
        )

    # Calculate total MD steps
    sampling_steps = sampling_configs * sampling_interval
    total_steps = equilibration_steps + sampling_steps

    # Start with static INCAR base
    incar_params = STATIC_INCAR_BASE.copy()

    # Apply MD modifications
    md_modifications = {
        # Core MD settings
        "IBRION": 0,  # Enable MD
        "NSW": total_steps,  # Total MD steps
        "POTIM": POTIM,  # Timestep in fs
        "MDALGO": 2,  # Nose-Hoover thermostat
        "RANDOM_SEED": random_seed,  # Random seed for independent trajectories
        # Temperature control - gentle ramp from 300K to target
        "TEBEG": ELEMENT_DATA[element]["init_temp"],
        "TEEND": ELEMENT_DATA[element]["target_temp"],  # Target temperature
        # Electronic structure adjustments for MD
        "ISMEAR": 0,  # Gaussian smearing (better with thermostat)
        # Output control
        "NBLOCK": sampling_interval,  # Write coordinates every sampling_interval steps
    }

    # Update INCAR with MD modifications
    incar_params.update(md_modifications)

    # Prepare simulation info
    simulation_info = {
        "element": element,
        "random_seed": random_seed,
        "target_temperature": ELEMENT_DATA[element]["target_temp"],
        "melting_point": ELEMENT_DATA[element]["melting_point"],
        "total_steps": total_steps,
        "equilibration_steps": equilibration_steps,
        "sampling_steps": sampling_steps,
        "sampling_configs": sampling_configs,
        "sampling_interval": sampling_interval,
        "total_time_ps": total_steps * 1.0 / 1000,  # Total time in ps
        "equilibration_time_ps": equilibration_steps * 1.0 / 1000,
        "sampling_time_ps": sampling_steps * 1.0 / 1000,
    }

    return incar_params, simulation_info


def create_multiple_runs(base_dir="e-supercells", num_seeds=10):
    """
    Create multiple MD run directories with different random seeds.

    Args:
        base_dir: Base directory to search for POSCAR files
        num_seeds: Number of different random seeds (default: 10)
    """
    import shutil

    from vsf.logging import setup_logging
    from vsf.vasp.input.files import VaspInputFiles

    setup_logging()
    base_dir = Path(base_dir)

    # Find POSCAR files
    poscar_paths = list(base_dir.rglob("**/POSCAR"))

    LOGGER.info(f"Found {len(poscar_paths)} POSCAR files")

    results = {}

    for poscar_path in poscar_paths:
        try:
            LOGGER.info(f"Processing: {poscar_path}")

            # Detect element from POSCAR
            element = detect_element_from_poscar(poscar_path)
            LOGGER.info(f"Detected element: {element}")

            # Get parent directory name for creating seed directories
            parent_dir_name = poscar_path.parent.name

            # Create multiple seed directories
            sim_info = None
            for seed in range(1, num_seeds + 1):
                # Create new directory name
                seed_dir_name = f"{parent_dir_name}_seed{seed}"
                seed_dir_path = base_dir / seed_dir_name

                # Create directory
                seed_dir_path.mkdir(exist_ok=True)

                # Copy POSCAR to new directory
                new_poscar_path = seed_dir_path / "POSCAR"
                shutil.copy2(poscar_path, new_poscar_path)

                # Generate MD INCAR parameters with specific seed
                md_incar_params, sim_info = create_md_incar_params(
                    element, random_seed=seed
                )

                # Create VaspInputFiles object for the new POSCAR
                vasp_inputs = VaspInputFiles(new_poscar_path)

                # Save POTCAR
                vasp_inputs.save_potcar()

                # Save MD INCAR with seed-specific parameters
                vasp_inputs.save_incar(
                    custom_incar_params=md_incar_params,
                    rewrite=True,
                )

                # Save KPOINTS
                create_gamma_kpoints(new_poscar_path)
                # vasp_inputs.save_kpoints(
                #     min_distance=28.0,
                #     include_gamma="auto",
                # )

                LOGGER.info(
                    f"✓ Created {seed_dir_name}: T={sim_info['target_temperature']}K, "
                    f"Steps={sim_info['total_steps']}, Seed={seed}"
                )

            # Store results for summary (using info from last seed, but it's the same for all)
            results[element] = sim_info

        except Exception as e:
            LOGGER.error(f"Failed to process {poscar_path}: {e}")

    return results


def print_simulation_summary(results):
    """Print a summary table of all simulations."""
    print("\n" + "=" * 70)
    print("MD SIMULATION SUMMARY")
    print("=" * 70)
    print(
        f"{'Element':<8} {'T_melt (K)':<12} {'T_target (K)':<14} {'Total Steps':<12} {'Time (ps)':<10}"
    )
    print("-" * 70)

    for element, info in results.items():
        print(
            f"{element:<8} {info['melting_point']:<12} {info['target_temperature']:<14} "
            f"{info['total_steps']:<12} {info['total_time_ps']:<10.1f}"
        )

    print(f"\nSampling: 100 configurations every {SAMPLING_INTERVAL} steps")
    print(f"Timestep: {POTIM} fs")
    print("Thermostat: Nose-Hoover")
    print("Random seeds: 1-10 (10 independent runs per element)")


def main():
    """Main workflow: create multiple MD runs with different seeds."""
    print("Creating MD inputs for liquid metal simulations with multiple seeds...")

    # Create multiple runs with different seeds
    results = create_multiple_runs(base_dir="e-supercells", num_seeds=10)

    if results:
        print_simulation_summary(results)
        print(
            f"\nCreated 10 independent runs for each element (total: {len(results) * 10} directories)"
        )
    else:
        print("No MD inputs were created. Check POSCAR files and directory structure.")


if __name__ == "__main__":
    main()
