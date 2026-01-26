"""
Simple plotting utilities for MD trajectory analysis.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


def plot_force_collapse(
    trajectory_json_path: Path, output_path: Optional[Path] = None
) -> None:
    """
    Plot max force vs simulation time with collapse annotation.

    Loads trajectory JSON, extracts max_forces and times, plots with vertical
    line marking collapse point (if detected).

    Args:
        trajectory_json_path: Path to trajectory.json
        output_path: Where to save plot. If None, displays to screen.
    """
    # Load trajectory
    with open(trajectory_json_path) as f:
        traj_data = json.load(f)

    times_ps = np.array(traj_data["trajectory_times_ps"])
    max_forces = np.array(traj_data["trajectory_max_forces"])
    collapse_time_ps = traj_data.get("collapse_time_ps")
    collapse_reason = traj_data.get("collapse_reason")

    # Debug: check dimensions
    LOGGER.debug(f"times_ps.shape = {times_ps.shape}")
    LOGGER.debug(f"max_forces.shape = {max_forces.shape}")
    LOGGER.debug(f"collapse_time_ps = {collapse_time_ps}")
    LOGGER.debug(f"collapse_reason = {collapse_reason}")
    LOGGER.debug(f"trajectory length = {len(traj_data['trajectory'])}")
    LOGGER.debug(f"energies length = {len(traj_data['trajectory_energies'])}")

    if len(times_ps) != len(max_forces):
        LOGGER.info(f"\nWARNING: Dimension mismatch!")
        LOGGER.info(f"  times_ps has {len(times_ps)} entries")
        LOGGER.info(f"  max_forces has {len(max_forces)} entries")
        LOGGER.info(f"  Difference: {len(times_ps) - len(max_forces)} entries")
        if collapse_time_ps is not None:
            LOGGER.info(f"  Collapsed at step (time {collapse_time_ps} ps)")
            LOGGER.info(
                f"  This suggests max_forces not saved during post-collapse steps"
            )

    # Handle mismatch by truncating to common length
    min_len = min(len(times_ps), len(max_forces))
    if len(times_ps) != len(max_forces):
        times_ps = times_ps[:min_len]
        max_forces = max_forces[:min_len]

    # Convert to fs
    times_fs = times_ps * 1000

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times_fs, max_forces, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Simulation Time [fs]", fontsize=12)
    ax.set_ylabel("Max Force [eV/Å]", fontsize=12)
    ax.set_title("Force Field Breakdown During MD", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Mark collapse if detected
    if collapse_time_ps is not None:
        collapse_time_fs = collapse_time_ps * 1000
        ax.axvline(
            collapse_time_fs, color="red", linestyle="--", linewidth=2, label="Collapse"
        )
        ax.legend(fontsize=11)

        # Add text annotation
        if collapse_reason:
            ax.text(
                collapse_time_fs,
                ax.get_ylim()[1] * 0.95,
                f"{collapse_reason}",
                fontsize=9,
                ha="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        LOGGER.info(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close(fig)
