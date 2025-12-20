import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

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
        max_count: Optional[int] = None,
    ) -> Optional[str]:
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


def parse_kpoint_scaling_calculations(base_dir: str) -> Dict[str, List[Dict]]:
    """
    Parse VASP k-point scaling calculations from directory structure.

    Args:
        base_dir: Base directory containing CPU-specific subdirectories (rrand_16, rrand_32, etc.)

    Returns:
        Dictionary with CPU counts as keys, containing lists of calculation data
        Each entry contains: {'structure_id': 'r0', 'kpoints': '222', 'TIME': X, 'cpu_count': 16, 'entry_dir': 'r0-222_'}
    """
    base_path = Path(base_dir)
    ripper = RipgrepRunner(show_progress=False)

    result = {}

    # Find all CPU directories (rrand_X format)
    cpu_dirs = [
        d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("rrand_")
    ]

    for cpu_dir in cpu_dirs:
        try:
            # Extract CPU count from directory name (rrand_16 -> 16)
            cpu_count_str = cpu_dir.name.replace("rrand_", "")
            cpu_count = int(cpu_count_str)

            if cpu_count not in result:
                result[cpu_count] = []

            # Find all calculation directories ending with "_"
            calc_dirs = [
                d for d in cpu_dir.iterdir() if d.is_dir() and d.name.endswith("_")
            ]

            for calc_dir in calc_dirs:
                try:
                    # Skip non-calculation directories
                    if calc_dir.name.startswith("0-"):
                        continue

                    # Parse structure and k-points from directory name (r0-222_ -> structure='r0', kpoints='222')
                    dir_name = calc_dir.name.rstrip("_")

                    if "-" not in dir_name:
                        print(f"Unexpected directory format: {calc_dir.name}")
                        continue

                    structure_id, kpoints_str = dir_name.split("-", 1)

                    # Validate k-points format (should be 3 digits)
                    if not kpoints_str.isdigit() or len(kpoints_str) != 3:
                        print(
                            f"Invalid k-points format in {calc_dir.name}: {kpoints_str}"
                        )
                        continue

                    # Find OUTCAR file
                    outcar_path = calc_dir / "OUTCAR"
                    if not outcar_path.exists():
                        print(f"OUTCAR not found in {calc_dir}")
                        continue

                    # Extract elapsed time
                    elapsed_search = ripper.search("Elapsed time", outcar_path)
                    if not elapsed_search:
                        print(f"Could not find elapsed time in {calc_dir}")
                        continue

                    # Parse elapsed time: "Elapsed time (sec):       13.880"
                    elapsed_line = elapsed_search.strip()
                    try:
                        time_value = float(elapsed_line.split(":")[1].strip())
                    except (IndexError, ValueError):
                        print(f"Could not parse elapsed time from {calc_dir}")
                        continue

                    # Create entry
                    entry = {
                        "structure_id": structure_id,
                        "kpoints": kpoints_str,
                        "TIME": time_value,
                        "cpu_count": cpu_count,
                        "entry_dir": calc_dir.name,
                    }

                    result[cpu_count].append(entry)

                except Exception as e:
                    print(f"Error processing {calc_dir}: {e}")
                    continue

        except ValueError:
            print(f"Could not parse CPU count from directory name: {cpu_dir.name}")
            continue
        except Exception as e:
            print(f"Error processing CPU directory {cpu_dir}: {e}")
            continue

    return result


def plot_kpoint_scaling(data_dict: Dict[str, List[Dict]]):
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
    fig, axes = plt.subplots(1, len(cpu_counts), figsize=(6 * len(cpu_counts), 6))
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


def plot_cpu_comparison(data_dict: Dict[str, List[Dict]]):
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

    fig, axes = plt.subplots(
        1, len(all_structures), figsize=(6 * len(all_structures), 6)
    )
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


def analyze_scaling_efficiency(data_dict: Dict[str, List[Dict]]):
    """
    Analyze and print scaling efficiency metrics.

    Args:
        data_dict: Dictionary from parse_kpoint_scaling_calculations
    """
    print("\n=== K-point Scaling Analysis ===\n")

    for cpu_count in sorted(data_dict.keys()):
        print(f"CPU Count: {cpu_count}")
        data = data_dict[cpu_count]

        if not data:
            print("  No data available\n")
            continue

        # Group by structure
        structure_groups = {}
        for entry in data:
            struct = entry["structure_id"]
            if struct not in structure_groups:
                structure_groups[struct] = []
            structure_groups[struct].append(entry)

        for struct, entries in structure_groups.items():
            entries.sort(key=lambda x: x["kpoints"])
            print(f"  Structure {struct}:")

            for entry in entries:
                print(f"    K-points {entry['kpoints']}: {entry['TIME']:.2f} sec")

            # Calculate scaling ratios
            if len(entries) > 1:
                base_time = entries[0]["TIME"]
                print(f"    Scaling ratios (relative to {entries[0]['kpoints']}):")
                for entry in entries[1:]:
                    ratio = entry["TIME"] / base_time
                    print(f"      {entry['kpoints']}: {ratio:.2f}x")
            print()


# Example usage:
if __name__ == "__main__":
    # Parse k-point scaling calculations
    data_dir = "data/"
    data = parse_kpoint_scaling_calculations(data_dir)

    # Generate plots
    plot_kpoint_scaling(data)
    plot_cpu_comparison(data)

    # Print analysis
    analyze_scaling_efficiency(data)
