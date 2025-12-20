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


def parse_vasp_calculations(
    base_dir: str, calc_source_name: str
) -> Dict[str, List[Dict]]:
    """
    Parse VASP calculations from directory structure.

    Args:
        base_dir: Base directory containing calculation directories
        calc_source_name: Name to identify this calculation source

    Returns:
        Dictionary with 'catalyst' and 'random' keys containing lists of calculation data
        Each entry contains: {'NCORE': X, 'KPAR': Y, 'TIME': Z, 'calc_source_name': name, 'entry_dir': dir_name}
    """
    base_path = Path(base_dir)
    ripper = RipgrepRunner(show_progress=False)

    result = {"catalyst": [], "random": []}

    # Find all directories ending with "_"
    entry_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.endswith("_")]

    for entry_dir in entry_dirs:
        try:
            # Determine category based on "--" in name
            category = "catalyst" if "--" in entry_dir.name else "random"

            # Find OUTCAR file
            outcar_path = entry_dir / "OUTCAR"
            if not outcar_path.exists():
                print(f"OUTCAR not found in {entry_dir}")
                continue

            # Extract elapsed time
            elapsed_search = ripper.search("Elapsed time", outcar_path)
            if not elapsed_search:
                print(f"Could not find elapsed time in {entry_dir}")
                continue

            # Parse elapsed time: "Elapsed time (sec):       13.880"
            elapsed_line = elapsed_search.strip()
            try:
                time_value = float(elapsed_line.split(":")[1].strip())
            except (IndexError, ValueError):
                print(f"Could not parse elapsed time from {entry_dir}")
                continue

            # Extract NCORE and KPAR
            distr_search = ripper.search("distr:", outcar_path)
            if not distr_search:
                print(f"Could not find distr line in {entry_dir}")
                continue

            # Parse distr line: "distr:  one band on NCORE=  16 cores,    1 groups"
            distr_line = distr_search.strip()
            try:
                # Extract NCORE (number before "cores")
                cores_part = distr_line.split("cores")[0]
                ncore_value = int(cores_part.split("=")[1].strip())

                # Extract KPAR (number before "groups")
                groups_part = distr_line.split("cores")[1]
                kpar_value = int(groups_part.split("groups")[0].strip().strip(","))
            except (IndexError, ValueError) as e:
                print(f"Could not parse NCORE/KPAR from {entry_dir}")
                print(f"  Full distr line: '{distr_line}'")
                try:
                    cores_part = distr_line.split("cores")[0]
                    print(f"  Cores part: '{cores_part}'")
                    ncore_part = cores_part.split("=")[1].strip()
                    print(f"  NCORE part: '{ncore_part}'")
                except:
                    print(f"  Failed to split cores part")
                try:
                    groups_part = distr_line.split("cores")[1]
                    print(f"  Groups part: '{groups_part}'")
                    kpar_part = groups_part.split("groups")[0].strip().strip(",")
                    print(f"  KPAR part: '{kpar_part}'")
                except:
                    print(f"  Failed to split groups part")
                print(f"  Error: {e}")
                continue

            # Create entry
            entry = {
                "NCORE": ncore_value,
                "KPAR": kpar_value,
                "TIME": time_value,
                "calc_source_name": calc_source_name,
                "entry_dir": entry_dir.name,
            }

            result[category].append(entry)

        except Exception as e:
            print(f"Error processing {entry_dir}: {e}")
            continue

    return result


def plot_vasp_timing(
    all_data_dicts: List[Dict[str, List[Dict]]],
    categories: List[str] = ["catalyst", "random"],
):
    """
    Plot timing data from multiple VASP calculation results.

    Args:
        all_data_dicts: List of dictionaries from parse_vasp_calculations
        categories: Categories to plot (default: ['catalyst', 'random'])
    """
    # Combine all data by category
    combined_data = {cat: [] for cat in categories}

    for data_dict in all_data_dicts:
        for category in categories:
            if category in data_dict:
                combined_data[category].extend(data_dict[category])

    # Create subplots
    fig, axes = plt.subplots(1, len(categories), figsize=(12, 6))
    if len(categories) == 1:
        axes = [axes]

    # Define markers for different NCORE/KPAR combinations
    markers = ["o", "s", "^", "v", "<", ">", "D", "p", "*", "h"]

    for i, category in enumerate(categories):
        ax = axes[i]
        data = combined_data[category]

        if not data:
            ax.set_title(f"{category.capitalize()} (No Data)")
            continue

        # Get all unique entry_dir names for color mapping
        all_entry_dirs = list(set(entry["entry_dir"] for entry in data))
        all_entry_dirs.sort()  # Sort for consistent colors

        # Create color map for entry_dirs
        colors = cm.viridis(np.linspace(0, 1, len(all_entry_dirs)))  # pyright: ignore
        entry_dir_colors = {name: colors[j] for j, name in enumerate(all_entry_dirs)}

        # Group by NCORE/KPAR combinations
        ncore_kpar_groups = {}
        for entry in data:
            key = (entry["NCORE"], entry["KPAR"])
            if key not in ncore_kpar_groups:
                ncore_kpar_groups[key] = []
            ncore_kpar_groups[key].append(entry)

        # Plot each NCORE/KPAR group
        for j, ((ncore, kpar), group) in enumerate(ncore_kpar_groups.items()):
            marker = markers[j % len(markers)]

            # Plot each entry with its own color based on entry_dir
            for entry in group:
                color = entry_dir_colors[entry["entry_dir"]]
                ax.scatter(
                    entry["calc_source_name"],
                    entry["TIME"],
                    marker=marker,
                    s=60,
                    alpha=0.7,
                    color=color,
                    label=f"NCORE={ncore}, KPAR={kpar}" if entry == group[0] else "",
                )

        ax.set_xlabel("calc_source_name")
        ax.set_ylabel("TIME (sec)")
        ax.set_title(f"{category.capitalize()} Calculations")

        # Create legend for NCORE/KPAR combinations
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys())

        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Parse calculations from different sources
    result1 = parse_vasp_calculations("ifort", "ifort")
    result2 = parse_vasp_calculations("ifx", "ifx")
    result3 = parse_vasp_calculations("native", "zhores_build_ifort")

    # Plot all results
    plot_vasp_timing([result1, result2, result3])
