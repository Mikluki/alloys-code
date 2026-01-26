import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from vsf.core.plot.utils import CHILL_COLORS, save_plot_to_file, set_plot_defaults

LOGGER = logging.getLogger(__name__)


def extract_hull_data(
    base_path: Path,
) -> Dict[str, Dict]:
    """
    Extract hull metrics data from directory structure.

    Scans base_path for directories matching pattern: {structure_id}_{volume_coeff}
    Parses JSON and loads POSCAR for each directory.

    Returns:
        {
            'Cu3Pd_mp-580357': {
                'potential_energy': {
                    0.8: {'VASP': -3.74, 'MACE_MPA_0': -3.74, ...},
                    1.0: {...},
                    1.2: {...}
                },
                'formation_energy': {...},
                'pressure': {
                    0.8: {'VASP': 0.04, ...},
                    1.0: {...},
                    1.2: {...}
                },
                'volume_fractions': {0.8: 0.96, 1.0: 1.0, 1.2: 1.04}
            },
            ...
        }
    """
    base_path = Path(base_path)
    data = {}
    volumes = {}  # Track volumes per structure for computing fractions

    # First pass: extract energies, stress, and volumes
    for dir_path in sorted(base_path.iterdir()):
        if not dir_path.is_dir():
            continue

        # Parse dir name: structure_id_volume_coeff
        parts = dir_path.name.rsplit("_", 1)
        if len(parts) != 2:
            continue

        structure_id, coeff_str = parts
        try:
            volume_coeff = float(coeff_str)
        except ValueError:
            continue

        # Load JSON
        json_path = dir_path / f"{dir_path.name}.json"
        with open(json_path) as f:
            file_data = json.load(f)

        # Load POSCAR for volume
        poscar_path = dir_path / "POSCAR"
        atoms = read(poscar_path)
        volume = atoms.get_volume()  # pyright: ignore

        # Initialize structure entry
        if structure_id not in data:
            data[structure_id] = {
                "potential_energy": {},
                "formation_energy": {},
                "pressure": {},
            }
            volumes[structure_id] = {}

        # Store volume
        volumes[structure_id][volume_coeff] = volume

        # Extract energies
        for energy_type in ["potential_energy", "formation_energy"]:
            methods_energies = {}
            energy_data = file_data.get(energy_type, {}).get("results", {})
            for method_name, method_data in energy_data.items():
                energy_value = method_data.get("value")
                if energy_value is not None:
                    methods_energies[method_name] = energy_value

            data[structure_id][energy_type][volume_coeff] = methods_energies

        # Extract pressure from stress
        methods_pressures = {}
        stress_data = file_data.get("stress_analyzer", {}).get("results", {})
        for method_name, method_info in stress_data.items():
            stress_array = method_info.get("stress_array")
            if stress_array is not None:
                # Pressure = -Tr(stress)/3, trace is first 3 elements
                trace = stress_array[0] + stress_array[1] + stress_array[2]
                pressure = -trace / 3
                methods_pressures[method_name] = pressure

        data[structure_id]["pressure"][volume_coeff] = methods_pressures

    # Second pass: compute volume fractions (relative to coeff 1.0)
    for structure_id in data:
        if 1.0 not in volumes[structure_id]:
            raise ValueError(f"Structure {structure_id} missing volume coefficient 1.0")

        ref_volume = volumes[structure_id][1.0]
        volume_fractions = {
            coeff: vol / ref_volume for coeff, vol in volumes[structure_id].items()
        }
        data[structure_id]["volume_fractions"] = volume_fractions
        data[structure_id]["volumes"] = volumes[structure_id]

    return data


def _birch_murnaghan_energy(
    volumes: np.ndarray, E_0: float, B: float, B_prime: float, V_0: float
) -> np.ndarray:
    """
    Birch-Murnaghan EOS energy (3rd order).

    Args:
        volumes: Array of volumes
        E_0: Equilibrium energy
        B: Bulk modulus
        B_prime: Pressure derivative of bulk modulus
        V_0: Equilibrium volume (fixed)

    Returns:
        Array of energies following BM equation
    """
    eta = np.power(volumes / V_0, 1.0 / 3.0)
    eta_sq = eta**2

    term = eta_sq - 1
    energy = E_0 + (9 * B * V_0 / 16) * (term**2) * (6 + B_prime * term - 4 * eta_sq)

    return energy


def fit_birch_murnaghan(
    volumes: np.ndarray, energies: np.ndarray, V_0: float
) -> Tuple[float, float, float, float]:
    """
    Fit Birch-Murnaghan EOS to energy-volume data.

    Args:
        volumes: Array of actual volumes
        energies: Array of energies
        V_0: Reference volume (fixed)

    Returns:
        (E_0, B, B_prime, residual_error)
    """
    # Initial guesses
    E_0_guess = energies.min()
    B_guess = 100  # GPa, typical value
    B_prime_guess = 4.0

    try:
        popt, _ = curve_fit(
            lambda V, E_0, B, B_p: _birch_murnaghan_energy(V, E_0, B, B_p, V_0),
            volumes,
            energies,
            p0=[E_0_guess, B_guess, B_prime_guess],
            maxfev=5000,
        )

        E_0_fit, B_fit, B_prime_fit = popt

        # Calculate residual error
        E_fit = _birch_murnaghan_energy(volumes, E_0_fit, B_fit, B_prime_fit, V_0)
        residual = np.sqrt(np.mean((energies - E_fit) ** 2))

        return E_0_fit, B_fit, B_prime_fit, residual
    except RuntimeError:
        # Fitting failed, return None values
        return None, None, None, None  # pyright: ignore


def _render_hull_plot(
    structure_id: str,
    structure_data: Dict[float, Dict[str, float]],
    energy_type: str,
) -> Figure:
    """
    Render hull plot for a single structure and energy type.

    Args:
        structure_id: Name of the structure
        structure_data: {0.8: {'VASP': -3.74, ...}, 1.0: {...}, ...}
        energy_type: 'potential_energy' or 'formation_energy'

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort volume coefficients
    sorted_coeffs = sorted(structure_data.keys())

    # Get all unique methods
    all_methods = set()
    for coeff_data in structure_data.values():
        all_methods.update(coeff_data.keys())
    sorted_methods = sorted(all_methods)

    # Plot each method
    colors = list(CHILL_COLORS)
    for method_idx, method_name in enumerate(sorted_methods):
        color = colors[method_idx % len(colors)]

        # Collect energies for this method across all coefficients
        coeffs_for_plot = []
        energies_for_plot = []
        for coeff in sorted_coeffs:
            if method_name in structure_data[coeff]:
                coeffs_for_plot.append(coeff)
                energies_for_plot.append(structure_data[coeff][method_name])

        if coeffs_for_plot:
            ax.plot(
                coeffs_for_plot,
                energies_for_plot,
                marker="+",
                label=method_name,
                color=color,
            )

    # Format label: convert potential_energy to "Potential Energy"
    label = energy_type.replace("_", " ").title()

    ax.set_xlabel("Volume Coefficient")
    ax.set_ylabel(f"{label} (eV)")
    ax.set_title(f"{structure_id} - {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_hull_metrics(base_path: Path, output_dir: Path) -> None:
    """
    Plot hull metrics for all structures and energy types.

    Extracts data from base_path, generates plots for both potential_energy and
    formation_energy per structure, saves to output_dir.

    Args:
        base_path: Directory containing structure subdirectories
        output_dir: Directory to save plots
    """
    set_plot_defaults()

    data = extract_hull_data(base_path)

    output_dir = Path(output_dir)

    for structure_id, structure_energies in data.items():
        for energy_type in ["potential_energy", "formation_energy"]:
            energy_data = structure_energies[energy_type]
            if not energy_data:
                continue

            fig = _render_hull_plot(structure_id, energy_data, energy_type)
            output_path = output_dir / f"{structure_id}_hull_{energy_type}.png"
            save_plot_to_file(fig, output_path, dpi=300)


def _render_pressure_plot(
    structure_id: str,
    pressure_data: Dict[float, Dict[str, float]],
    volume_fractions: Dict[float, float],
) -> Figure:
    """
    Render pressure vs volume fraction plot for a single structure.

    Args:
        structure_id: Name of the structure
        pressure_data: {0.8: {'VASP': 0.04, ...}, 1.0: {...}, ...}
        volume_fractions: {0.8: 0.96, 1.0: 1.0, 1.2: 1.04}

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort volume coefficients
    sorted_coeffs = sorted(pressure_data.keys())

    # Get all unique methods
    all_methods = set()
    for coeff_data in pressure_data.values():
        all_methods.update(coeff_data.keys())
    sorted_methods = sorted(all_methods)

    # Plot each method
    colors = list(CHILL_COLORS)
    for method_idx, method_name in enumerate(sorted_methods):
        color = colors[method_idx % len(colors)]

        # Collect pressures and volume fractions for this method
        vol_fracs_for_plot = []
        pressures_for_plot = []
        for coeff in sorted_coeffs:
            if method_name in pressure_data[coeff]:
                vol_fracs_for_plot.append(volume_fractions[coeff])
                pressures_for_plot.append(pressure_data[coeff][method_name])

        if vol_fracs_for_plot:
            ax.plot(
                vol_fracs_for_plot,
                pressures_for_plot,
                marker="x",
                label=method_name,
                color=color,
            )

    ax.set_xlabel("Volume Fraction")
    ax.set_ylabel("Pressure (eV/Ų)")
    ax.set_title(f"{structure_id} - Pressure vs Volume Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_pressure_metrics(base_path: Path, output_dir: Path) -> None:
    """
    Plot pressure vs volume fraction for all structures.

    Extracts data from base_path, generates one plot per structure,
    saves to output_dir.

    Args:
        base_path: Directory containing structure subdirectories
        output_dir: Directory to save plots
    """
    set_plot_defaults(line_width=1, marker_size=10)

    data = extract_hull_data(base_path)

    output_dir = Path(output_dir)

    for structure_id, structure_data in data.items():
        pressure_data = structure_data["pressure"]
        volume_fractions = structure_data["volume_fractions"]

        if not pressure_data:
            continue

        fig = _render_pressure_plot(structure_id, pressure_data, volume_fractions)
        output_path = output_dir / f"{structure_id}_pressure.png"
        save_plot_to_file(fig, output_path, dpi=300)


def _render_eos_benchmark_plot(
    structure_id: str,
    energies_data: Dict[float, Dict[str, float]],
    volumes: Dict[float, float],
) -> Figure:
    """
    Render normalized Birch-Murnaghan EOS benchmark plot.

    Plots reduced normalized energy (E - E_0) / (B*V_0) vs η = (V/V_0)^(1/3)
    for all methods with fitted BM curves.

    Args:
        structure_id: Name of the structure
        energies_data: {0.8: {'VASP': -3.74, ...}, 1.0: {...}, ...}
        volumes: {0.8: 123.45, 1.0: 150.0, ...}

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort volume coefficients
    sorted_coeffs = sorted(energies_data.keys())

    # Get all unique methods
    all_methods = set()
    for coeff_data in energies_data.values():
        all_methods.update(coeff_data.keys())
    sorted_methods = sorted(all_methods)

    # Reference volume (at coeff 1.0)
    V_0 = volumes[1.0]

    # Plot each method
    colors = list(CHILL_COLORS)
    for method_idx, method_name in enumerate(sorted_methods):
        color = colors[method_idx % len(colors)]

        # Collect volumes and energies for this method
        volumes_for_method = []
        energies_for_method = []
        for coeff in sorted_coeffs:
            if method_name in energies_data[coeff]:
                volumes_for_method.append(volumes[coeff])
                energies_for_method.append(energies_data[coeff][method_name])

        if len(volumes_for_method) < 3:
            continue

        # Fit Birch-Murnaghan
        volumes_array = np.array(volumes_for_method)
        energies_array = np.array(energies_for_method)
        E_0, B, B_prime, residual = fit_birch_murnaghan(
            volumes_array, energies_array, V_0
        )

        if E_0 is None:
            continue

        # Compute eta and normalized reduced energy for data points
        eta_data = np.power(volumes_array / V_0, 1.0 / 3.0)
        reduced_energy_data = (energies_array - E_0) / (B * V_0)

        # Generate fitted curve
        V_fit = np.linspace(volumes_array.min(), volumes_array.max(), 100)
        E_fit = _birch_murnaghan_energy(V_fit, E_0, B, B_prime, V_0)
        eta_fit = np.power(V_fit / V_0, 1.0 / 3.0)
        reduced_energy_fit = (E_fit - E_0) / (B * V_0)

        # Plot data points as scatter
        ax.scatter(
            eta_data, reduced_energy_data, color=color, s=50, zorder=3, marker="+"
        )

        # Plot fitted curve
        ax.plot(
            eta_fit, reduced_energy_fit, color=color, linewidth=2.0, label=method_name
        )

    ax.set_xlabel("η = (V/V₀)^(1/3)")
    ax.set_ylabel("(E - E₀) / (B·V₀)")
    ax.set_title(f"{structure_id} - EOS Benchmark (Normalized Reduced Energy)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_eos_benchmark(base_path: Path, output_dir: Path) -> None:
    """
    Plot Birch-Murnaghan EOS benchmark for all structures.

    Fits BM EOS to each method, plots normalized reduced energy curves.

    Args:
        base_path: Directory containing structure subdirectories
        output_dir: Directory to save plots
    """
    set_plot_defaults(line_width=2.0, marker_size=8)

    data = extract_hull_data(base_path)

    output_dir = Path(output_dir)

    for structure_id, structure_data in data.items():
        energies_data = structure_data["potential_energy"]
        volumes = structure_data["volumes"]

        if not energies_data:
            continue

        fig = _render_eos_benchmark_plot(structure_id, energies_data, volumes)
        output_path = output_dir / f"{structure_id}_eos_benchmark.png"
        save_plot_to_file(fig, output_path, dpi=300)


def _render_eos_benchmark_by_method_plot(
    method_name: str,
    all_structures_data: Dict[str, Dict],
) -> Figure:
    """
    Render normalized BM EOS benchmark plot for all structures, single method.

    Shows how a single method (VASP, MACE, ORB, etc.) performs across different
    structures. Color-coded by structure.

    Args:
        method_name: Name of the method (e.g., 'VASP', 'MACE_MPA_0')
        all_structures_data: Full extracted data dict

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Assign colors to structures
    structure_ids = sorted(all_structures_data.keys())
    cmap = plt.cm.tab20  # pyright: ignore
    colors_map = {
        struct_id: cmap(i / max(len(structure_ids), 1))
        for i, struct_id in enumerate(structure_ids)
    }

    for structure_id, structure_data in all_structures_data.items():
        energies_data = structure_data["potential_energy"]
        volumes = structure_data["volumes"]

        if not energies_data:
            continue

        # Sort volume coefficients
        sorted_coeffs = sorted(energies_data.keys())

        # Collect volumes and energies for this method
        volumes_for_method = []
        energies_for_method = []
        for coeff in sorted_coeffs:
            if method_name in energies_data[coeff]:
                volumes_for_method.append(volumes[coeff])
                energies_for_method.append(energies_data[coeff][method_name])

        if len(volumes_for_method) < 3:
            continue

        # Reference volume for this structure
        V_0 = volumes[1.0]

        # Fit Birch-Murnaghan
        volumes_array = np.array(volumes_for_method)
        energies_array = np.array(energies_for_method)
        E_0, B, B_prime, residual = fit_birch_murnaghan(
            volumes_array, energies_array, V_0
        )

        if E_0 is None:
            continue

        # Generate fitted curve
        V_fit = np.linspace(volumes_array.min(), volumes_array.max(), 100)
        E_fit = _birch_murnaghan_energy(V_fit, E_0, B, B_prime, V_0)
        eta_fit = np.power(V_fit / V_0, 1.0 / 3.0)
        reduced_energy_fit = (E_fit - E_0) / (B * V_0)

        color = colors_map[structure_id]

        # Plot fitted curve
        ax.plot(
            eta_fit,
            reduced_energy_fit,
            color=color,
            linewidth=2.0,
            label=structure_id,
        )

        # Plot data points as scatter
        eta_data = np.power(volumes_array / V_0, 1.0 / 3.0)
        reduced_energy_data = (energies_array - E_0) / (B * V_0)
        ax.scatter(eta_data, reduced_energy_data, color=color, s=50, zorder=3)

    ax.set_xlabel("η = (V/V₀)^(1/3)")
    ax.set_ylabel("(E - E₀) / (B·V₀)")
    ax.set_title(f"EOS Benchmark - {method_name} (All Structures)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    return fig


def plot_eos_benchmark_by_method(base_path: Path, output_dir: Path) -> None:
    """
    Plot Birch-Murnaghan EOS benchmark grouped by method.

    Creates one plot per method showing all structures, color-coded by structure.
    Allows visual assessment of method robustness across different materials.

    Args:
        base_path: Directory containing structure subdirectories
        output_dir: Directory to save plots
    """
    set_plot_defaults(line_width=2.0, marker_size=8)

    data = extract_hull_data(base_path)

    output_dir = Path(output_dir)

    # Collect all methods across all structures
    all_methods = set()
    for structure_data in data.values():
        energies_data = structure_data["potential_energy"]
        for coeff_data in energies_data.values():
            all_methods.update(coeff_data.keys())

    # Plot for each method
    for method_name in sorted(all_methods):
        fig = _render_eos_benchmark_by_method_plot(method_name, data)
        output_path = output_dir / f"{method_name}_eos_benchmark.png"
        save_plot_to_file(fig, output_path, dpi=300)
