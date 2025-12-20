import re
import subprocess
from collections import defaultdict
from pathlib import Path

# from pprint import pp
from typing import Dict, List, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class RipgrepRunner:
    """Fast wrapper around ripgrep for searching files."""

    def __init__(self, show_progress: bool = True):
        """Initialize RipgrepRunner."""
        self.show_progress = show_progress

    def search(
        self,
        pattern: str,
        file_path: Union[str, Path],
        context_lines: int = 0,
        max_count: int | None = None,
    ) -> str | None:
        """Search for a pattern in a file using ripgrep."""
        file_path = Path(file_path)

        if not file_path.exists():
            if self.show_progress:
                print(f"File not found: {file_path}")
            return None

        # Use -e flag to specify pattern, especially for patterns starting with "-"
        cmd = ["rg", "-e", pattern]

        # Add context lines if needed
        if context_lines > 0:
            cmd.extend(["-A", str(context_lines)])

        # Add max count if needed
        if max_count is not None:
            cmd.extend(["-m", str(max_count)])

        # Add file path
        cmd.append(str(file_path))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            # Return None if pattern not found
            if result.returncode == 1:
                return None

            # Check for errors
            if result.returncode not in [0, 1]:  # 0=match found, 1=no match
                if self.show_progress:
                    print(f"Error running ripgrep on {file_path}: {result.stderr}")
                return None

            return result.stdout if result.stdout else None

        except Exception as e:
            if self.show_progress:
                print(f"Exception running ripgrep on {file_path}: {e}")
            return None


def extract_value_from_file(
    ripper: RipgrepRunner,
    file_path: Path,
    search_string: str,
    parser_func,
):
    """
    Extract and parse a value from a file.

    Args:
        ripper: RipgrepRunner instance
        file_path: Path to the file to search
        search_string: String to search for in the file
        parser_func: Function that takes the found line and returns the parsed value

    Returns:
        Parsed value or None if extraction/parsing fails
    """
    search_result = ripper.search(search_string, file_path)
    calc_dir = file_path.parents[0]
    if not search_result:
        print(f"Could not find '{search_string}' in {calc_dir}")
        return None

    line = search_result.strip()
    try:
        return parser_func(line)
    except (IndexError, ValueError, AttributeError) as e:
        print(f"Could not parse '{search_string}' from {calc_dir}: {e}")
        return None


def extract_cpu_count(dir_name: str) -> int | None:
    """
    Extract CPU count from directory name.

    Args:
        dir_name: Directory name like 'r0-222-ncpu16_' or 'r1-333_'

    Returns:
        CPU count as integer or None if not found
    """
    match = re.search(r"ncpu(\d+)", dir_name)
    if match:
        return int(match.group(1))
    return None


def extract_kpoints_count(kpoints_path: Path) -> int | None:
    """
    Read second line of KPOINTS file as integer.

    Args:
        kpoints_path: Path to KPOINTS file

    Returns:
        Number of k-points as integer, or None if extraction fails
    """
    calc_dir = kpoints_path.parents[0]
    try:
        with open(kpoints_path, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                print(f"KPOINTS file has fewer than 2 lines in {calc_dir}")
                return None
            return int(lines[1].strip())
    except (ValueError, IOError) as e:
        print(f"Could not parse KPOINTS file from {calc_dir}: {e}")
        return None


# Parser functions for different formats
def parse_elapsed_time(line):
    """Parse: 'Elapsed time (sec):       13.880'"""
    return float(line.split(":")[1].strip())


def parse_toten_energy(line):
    """Parse: 'free  energy   TOTEN  =      -122.81632474 eV'"""
    return float(line.split("=")[1].strip().split()[0])


def parse_n_atoms(line):
    """Parse: 'number of dos      NEDOS =    301   number of ions     NIONS =     16'"""
    return float(line.split("=")[-1].strip().split()[0])


def parse_mindistance(line):
    """Parse: 'MINDISTANCE=25.0'"""
    return float(line.split("=")[1].strip())


def parse_data_dirs(
    calc_dirs: list[Path], extract_min_distance: bool = True
) -> List[Dict]:
    """
    Parse VASP calculations from flat directory structure.

    Args:
        calc_dirs: list of dirs containing calculation subdirectories ending with "_"
        extract_min_distance: Whether to extract min_distance from PRECALC file (default: True)

    Returns:
        List of calculation data dictionaries
        Each entry contains: {'TIME': X, 'total_free_energy': Y, 'n_atoms': Z, 'cpu_count': N,
                             'entry_dir': 'dir_name', 'kpoints_count': K, 'min_distance': M}
    """
    ripper = RipgrepRunner(show_progress=False)

    result = []

    # Find all calculation directories ending with "_"
    for d in calc_dirs:
        if not d.is_dir():
            raise ValueError(f"Path `{d}` is not a directory")
        if not d.name.endswith("_"):
            raise ValueError(f"Directory `{d.name}` does not end with '_'")

    for calc_dir in calc_dirs:
        try:
            # Extract optional CPU count from directory name
            cpu_count = extract_cpu_count(calc_dir.name)

            # Find required files
            outcar_path = calc_dir / "OUTCAR"
            kpoints_path = calc_dir / "KPOINTS"
            precalc_path = calc_dir / "PRECALC"

            # Check OUTCAR exists
            if not outcar_path.exists():
                print(f"OUTCAR not found in {calc_dir}")
                continue

            # Check KPOINTS exists
            if not kpoints_path.exists():
                print(f"KPOINTS not found in {calc_dir}")
                continue

            # Check PRECALC exists (if required)
            if extract_min_distance and not precalc_path.exists():
                print(f"PRECALC not found in {calc_dir}")
                continue

            # Extract elapsed time
            time_value = extract_value_from_file(
                ripper,
                outcar_path,
                search_string="Elapsed time",
                parser_func=parse_elapsed_time,
            )
            if time_value is None:
                continue

            # Extract total free energy
            total_free_e_value = extract_value_from_file(
                ripper,
                outcar_path,
                search_string="free  energy   TOTEN",
                parser_func=parse_toten_energy,
            )
            if total_free_e_value is None:
                continue

            # Extract n atoms
            n_atoms_value = extract_value_from_file(
                ripper,
                outcar_path,
                search_string="number of ions     NIONS",
                parser_func=parse_n_atoms,
            )
            if n_atoms_value is None:
                continue

            # Extract kpoints count (always required)
            kpoints_count = extract_kpoints_count(kpoints_path)
            if kpoints_count is None:
                continue

            # Extract min distance (if required)
            min_distance = None
            if extract_min_distance:
                min_distance = extract_value_from_file(
                    ripper,
                    precalc_path,
                    search_string="MINDISTANCE",
                    parser_func=parse_mindistance,
                )
                if min_distance is None:
                    continue

            # Create entry
            entry = {
                "TIME": time_value,
                "total_free_energy": total_free_e_value,
                "free_energy_per_atom": total_free_e_value / kpoints_count,
                "n_atoms": n_atoms_value,
                "cpu_count": cpu_count,
                "entry_dir": calc_dir.name,
                "kpoints_count": kpoints_count,
            }

            # Add min_distance if extracted
            if extract_min_distance:
                entry["min_distance"] = min_distance

            result.append(entry)

        except Exception as e:
            print(f"Error processing {calc_dir}: {e}")
            continue

    return result


def analyze_kpoints_convergence(data: List[Dict]) -> Dict:
    """
    Analyze k-points convergence by sorting entries and calculating energy differences.

    Args:
        data: List of calculation dictionaries from parse_data_dirs()

    Returns:
        Dictionary containing:
        - 'sorted_data': Data sorted by kpoints_count
        - 'convergence_analysis': List of convergence steps with energy differences
        - 'summary': Convergence summary statistics
    """

    # Sort data by kpoints_count
    sorted_data = sorted(data, key=lambda x: x["kpoints_count"])

    # Calculate energy differences between successive k-point grids
    convergence_steps = []

    for i in range(1, len(sorted_data)):
        prev_calc = sorted_data[i - 1]
        curr_calc = sorted_data[i]

        # Energy difference (current - previous)
        energy_diff = curr_calc["total_free_energy"] - prev_calc["total_free_energy"]

        # Energy difference per atom
        energy_diff_per_atom = energy_diff / curr_calc["n_atoms"]

        # K-points increase
        kpoints_increase = curr_calc["kpoints_count"] - prev_calc["kpoints_count"]

        convergence_step = {
            "from_kpoints": prev_calc["kpoints_count"],
            "to_kpoints": curr_calc["kpoints_count"],
            "kpoints_increase": kpoints_increase,
            "energy_difference": energy_diff,
            "energy_diff_per_atom": energy_diff_per_atom,
            "energy_diff_abs": abs(energy_diff),
            "energy_diff_per_atom_abs": abs(energy_diff_per_atom),
            "from_dir": prev_calc["entry_dir"],
            "to_dir": curr_calc["entry_dir"],
            "from_time": prev_calc["TIME"],
            "to_time": curr_calc["TIME"],
            "converged_within_1meV": abs(energy_diff_per_atom)
            < 0.001,  # 1 meV threshold
        }

        convergence_steps.append(convergence_step)

    # Summary statistics
    if convergence_steps:
        max_energy_diff = max(step["energy_diff_abs"] for step in convergence_steps)
        max_energy_diff_per_atom = max(
            step["energy_diff_per_atom_abs"] for step in convergence_steps
        )

        # Find largest energy change step
        largest_change_step = max(
            convergence_steps, key=lambda x: x["energy_diff_per_atom_abs"]
        )

        # Check if converged (last step < 1 meV/atom)
        is_converged = convergence_steps[-1]["converged_within_1meV"]

        summary = {
            "total_kpoints_range": (
                sorted_data[0]["kpoints_count"],
                sorted_data[-1]["kpoints_count"],
            ),
            "total_energy_range": (
                sorted_data[0]["total_free_energy"],
                sorted_data[-1]["total_free_energy"],
            ),
            "max_energy_difference": max_energy_diff,
            "max_energy_diff_per_atom": max_energy_diff_per_atom,
            "largest_change_step": {
                "kpoints_range": (
                    largest_change_step["from_kpoints"],
                    largest_change_step["to_kpoints"],
                ),
                "energy_diff_per_atom": largest_change_step["energy_diff_per_atom"],
            },
            "is_converged_1meV": is_converged,
            "final_convergence_step": {
                "energy_diff_per_atom": convergence_steps[-1]["energy_diff_per_atom"],
                "energy_diff_per_atom_abs": convergence_steps[-1][
                    "energy_diff_per_atom_abs"
                ],
            },
        }
    else:
        summary = {"error": "Not enough data points for convergence analysis"}

    return {
        "sorted_data": sorted_data,
        "convergence_analysis": convergence_steps,
        "summary": summary,
    }


def print_convergence_analysis(analysis: Dict):
    """
    Pretty print the convergence analysis results.

    Args:
        analysis: Result dictionary from analyze_kpoints_convergence()
    """
    print("=" * 100)
    print("K-POINTS CONVERGENCE ANALYSIS (Energy per Atom)")
    print("=" * 100)

    if "error" in analysis["summary"]:
        print(f"Error: {analysis['summary']['error']}")
        return

    # Print sorted data
    print("\nSorted calculations by k-points:")
    print("-" * 100)
    for calc in analysis["sorted_data"]:
        energy_per_atom = calc["total_free_energy"] / calc["n_atoms"]
        min_dist = calc.get("min_distance", "N/A")
        print(
            f"K-points: {calc['kpoints_count']:3d} | "
            f"Min-dist: {min_dist:>6} | "
            f"Time: {calc['TIME']:8.1f}s | "
            f"Energy/atom: {energy_per_atom:12.6f} eV | "
            f"Dir: {calc['entry_dir']}"
        )

    # Print convergence steps
    print("\nConvergence analysis (successive differences):")
    print("-" * 100)
    print(
        f"{'From':>4} {'To':>4} {'From Time (s)':>14} {'To Time (s)':>12} {'|ΔE/atom| (meV)':>18} {'Converged':>12}"
    )
    print("-" * 100)

    for step in analysis["convergence_analysis"]:
        energy_diff_meV = step["energy_diff_per_atom_abs"] * 1000  # Convert to meV
        converged_symbol = "✓" if step["converged_within_1meV"] else "✗"

        print(
            f"{step['from_kpoints']:4d} {step['to_kpoints']:4d} "
            f"{step['from_time']:14.0f} "
            f"{step['to_time']:12.0f} "
            f"{energy_diff_meV:18.2f} "
            f"{converged_symbol:>12}"
        )

    # Print summary
    summary = analysis["summary"]
    print("\nSummary:")
    print("-" * 100)
    print(
        f"K-points range: {summary['total_kpoints_range'][0]} → {summary['total_kpoints_range'][1]}"
    )
    print(
        f"Max energy diff/atom: {summary['max_energy_diff_per_atom']:.6f} eV ({summary['max_energy_diff_per_atom']*1000:.1f} meV)"
    )

    largest = summary["largest_change_step"]
    print(
        f"Largest change: {largest['kpoints_range'][0]}→{largest['kpoints_range'][1]} k-points "
        f"({largest['energy_diff_per_atom']*1000:.1f} meV/atom)"
    )

    final = summary["final_convergence_step"]
    print(f"Final step: {final['energy_diff_per_atom_abs']*1000:.1f} meV/atom")
    print(
        f"Converged (< 1 meV/atom): {'YES' if summary['is_converged_1meV'] else 'NO'}"
    )


def plot_kpoints_grouped_by_cpu(data_dict: Dict[str, List[Dict]]):
    """
    Plot k-point scaling analysis showing timing vs k-point density.

    Args:
        data_dict: Dictionary from parse_kpoint_scaling_calculations
    """
    if not data_dict:
        print("No data to plot")
        return

    # Create figure with subplots for each CPU count
    cpu_counts = sorted(data_dict.keys())
    _, axes = plt.subplots(1, len(cpu_counts), figsize=(6 * len(cpu_counts), 6))
    if len(cpu_counts) == 1:
        axes = [axes]

    # Define colors and markers for different structures
    structures = set()
    for cpu_data in data_dict.values():
        structures.update(entry["structure_id"] for entry in cpu_data)

    structures = sorted(structures)
    colors = cm.Set1(np.linspace(0, 1, len(structures)))  # pyright: ignore
    markers = ["o", "s", "^", "v", "<", ">", "D", "p"]

    structure_styles = {
        struct: {"color": colors[i], "marker": markers[i % len(markers)]}
        for i, struct in enumerate(structures)
    }

    for i, cpu_count in enumerate(cpu_counts):
        ax = axes[i]
        data = data_dict[cpu_count]

        if not data:
            ax.set_title(f"{cpu_count} CPUs (No Data)")
            continue

        # Group by structure
        structure_groups = {}
        for entry in data:
            struct = entry["structure_id"]
            if struct not in structure_groups:
                structure_groups[struct] = []
            structure_groups[struct].append(entry)

        # Plot each structure
        for struct, entries in structure_groups.items():
            # Sort by k-points for proper line connections
            entries.sort(key=lambda x: x["kpoints"])

            kpoints = [entry["kpoints"] for entry in entries]
            times = [entry["TIME"] for entry in entries]

            style = structure_styles[struct]

            # Plot line and points
            ax.plot(
                kpoints,
                times,
                color=style["color"],
                marker=style["marker"],
                markersize=8,
                linewidth=2,
                alpha=0.7,
                label=f"Structure {struct}",
            )

        ax.set_xlabel("K-point Density")
        ax.set_ylabel("Elapsed Time (sec)")
        ax.set_title(f"{cpu_count} CPUs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set x-axis to show k-point values clearly
        ax.set_xticks([entry["kpoints"] for entry in data])

    plt.suptitle("K-point Scaling Analysis", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_kpoints_grouped_by_structure(data_dict: Dict[str, List[Dict]]):
    """
    Plot CPU efficiency comparison across different k-point densities.

    Args:
        data_dict: Dictionary from parse_kpoint_scaling_calculations
    """
    if len(data_dict) < 2:
        print("Need at least 2 CPU configurations for comparison")
        return

    # Get all unique k-point densities and structures
    all_kpoints = set()
    all_structures = set()

    for cpu_data in data_dict.values():
        all_kpoints.update(entry["kpoints"] for entry in cpu_data)
        all_structures.update(entry["structure_id"] for entry in cpu_data)

    all_kpoints = sorted(all_kpoints)
    all_structures = sorted(all_structures)

    _, axes = plt.subplots(1, len(all_structures), figsize=(6 * len(all_structures), 6))
    if len(all_structures) == 1:
        axes = [axes]

    # Colors for different CPU counts
    cpu_counts = sorted(data_dict.keys())
    colors = cm.viridis(np.linspace(0, 1, len(cpu_counts)))  # pyright: ignore
    cpu_colors = {cpu: colors[i] for i, cpu in enumerate(cpu_counts)}

    for i, structure in enumerate(all_structures):
        ax = axes[i]

        for cpu_count in cpu_counts:
            # Get data for this CPU count and structure
            cpu_data = data_dict[cpu_count]
            structure_data = [
                entry for entry in cpu_data if entry["structure_id"] == structure
            ]

            if not structure_data:
                continue

            # Sort by k-points
            structure_data.sort(key=lambda x: x["kpoints"])

            kpoints = [entry["kpoints"] for entry in structure_data]
            times = [entry["TIME"] for entry in structure_data]

            ax.plot(
                kpoints,
                times,
                color=cpu_colors[cpu_count],
                marker="o",
                markersize=8,
                linewidth=2,
                label=f"{cpu_count} CPUs",
            )

        ax.set_xlabel("K-point Density")
        ax.set_ylabel("Elapsed Time (sec)")
        ax.set_title(f"Structure {structure}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_kpoints)

    plt.suptitle("CPU Count Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()


def find_calc_dirs(base_dir: str) -> List[Path]:
    """
    Find all calculation directories ending with "_" in base directory.

    Args:
        base_dir: Base directory to search

    Returns:
        List of Path objects for calculation directories
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory `{base_dir}` does not exist")

    calc_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.endswith("_")]

    if not calc_dirs:
        raise ValueError(
            f"No calculation directories (ending with '_') found in {base_dir}"
        )

    return calc_dirs


def group_dirs_by_prefix(
    dirs: List[Path], prefix_pattern: str
) -> Dict[str, List[Path]]:
    """
    Group directories by a regex prefix pattern.

    Args:
        dirs: List of directory Path objects
        prefix_pattern: Regex pattern to extract prefix (should have one capture group)
                       e.g., r"(r\\d+)" for "r0", "r1", etc.

    Returns:
        Dictionary with prefixes as keys and lists of matching directories as values

    Example:
        dirs = [Path("r0_md_15.0_"), Path("r0_md_40.0_"), Path("r1_md_15.0_")]
        pattern = r"(r\\d+)"
        result = {"r0": [Path("r0_md_15.0_"), Path("r0_md_40.0_")],
                 "r1": [Path("r1_md_15.0_")]}
    """
    grouped = defaultdict(list)
    unmatched = []

    for d in dirs:
        match = re.match(prefix_pattern, d.name)
        if match:
            prefix = match.group(1)
            grouped[prefix].append(d)
        else:
            unmatched.append(d)

    if unmatched:
        print(
            f"Warning: {len(unmatched)} directories didn't match pattern '{prefix_pattern}':"
        )
        for d in unmatched:
            print(f"  - {d.name}")

    # Convert defaultdict to regular dict and sort directories within each group
    result = {}
    for prefix, dir_list in grouped.items():
        result[prefix] = sorted(dir_list, key=lambda x: x.name)

    return result


def group_dirs_by_structure_id(dirs: List[Path]) -> Dict[str, List[Path]]:
    """
    Convenience function to group directories by structure ID (r0, r1, r2, etc.).

    Args:
        dirs: List of directory Path objects

    Returns:
        Dictionary with structure IDs as keys and lists of directories as values
    """
    return group_dirs_by_prefix(dirs, r"(r\d+)")


# Example usage:
if __name__ == "__main__":
    # Parse k-point scaling calculations
    data_dir = "run/"

    # Get all calculation directories
    calc_dirs = find_calc_dirs(data_dir)

    # Group by structure ID
    grouped_dirs = group_dirs_by_structure_id(calc_dirs)

    grouped_data = {}
    for struct_id, dirs in grouped_dirs.items():
        print(f"Processing {len(dirs)} calculations for structure {struct_id}...")
        grouped_data[struct_id] = parse_data_dirs(dirs)

        data = analyze_kpoints_convergence(grouped_data[struct_id])

        # Print results
        print_convergence_analysis(data)
