import logging
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read

from .md_config import AnalysisConfig, VASPEquilibrationConfig

LOGGER = logging.getLogger(__name__)


# ============================================================================
# VASPTrajectoryAnalyzer
# ============================================================================


class VASPTrajectoryAnalyzer:
    """Parse and analyze VASP MD trajectories."""

    def __init__(self):
        pass

    def load_trajectory(self, vasp_dir: Path) -> dict:
        """
        Load VASP trajectory from directory.

        Expects:
        - vasp_dir/XDATCAR: atomic positions trajectory
        - vasp_dir/OUTCAR: optional, for temperature and energy extraction

        Returns:
            {
                "frames": list[Atoms],
                "temperatures": list[float],
                "energies": list[float],
                "times_ps": list[float],
            }
        """
        xdatcar_path = vasp_dir / "XDATCAR"
        outcar_path = vasp_dir / "OUTCAR"

        # Load trajectory from XDATCAR (required) or OUTCAR (fallback)
        if xdatcar_path.exists():
            LOGGER.info(f"Loading VASP trajectory from {xdatcar_path}")
            frames = ase_read(str(xdatcar_path), index=":")
            if not isinstance(frames, list):
                frames = [frames]
        elif outcar_path.exists():
            LOGGER.info(f"Loading VASP trajectory from {outcar_path}")
            frames = ase_read(str(outcar_path), index=":")
            if not isinstance(frames, list):
                frames = [frames]
        else:
            raise FileNotFoundError(f"Neither XDATCAR nor OUTCAR found in {vasp_dir}")

        LOGGER.info(f"Loaded {len(frames)} frames from VASP trajectory")

        # Extract temperatures and energies from OUTCAR if available
        if outcar_path.exists():
            temperatures, energies, times_ps = self._extract_md_data(
                outcar_path, frames
            )
        else:
            # Fallback: use defaults
            LOGGER.warning(
                "OUTCAR not found; using default temperature (1400 K) and frame energies"
            )
            md_timestep_fs = 2.0  # Standard default
            times_ps = [i * md_timestep_fs / 1000 for i in range(len(frames))]
            temperatures = [1400.0] * len(frames)
            energies = [frame.get_potential_energy() for frame in frames]

        return {
            "frames": frames,
            "temperatures": temperatures,
            "energies": energies,
            "times_ps": times_ps,
        }

    def _extract_md_data(
        self, outcar_path: Path, frames: list[Atoms]
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Extract temperature and energy time series from OUTCAR.

        Returns:
            (temperatures, energies, times_ps)
        """
        temperatures = []
        energies = []
        times_ps = []

        with open(outcar_path) as f:
            lines = f.readlines()

        # Simple parsing: look for MD step information
        # This is a minimal implementation; adjust based on your OUTCAR format
        md_step = 0
        for i, line in enumerate(lines):
            # Look for TOTEN (total energy) lines in MD
            if "TOTEN" in line and "=" in line:
                try:
                    energy = float(line.split("=")[1].split()[0])
                    energies.append(energy)
                    times_ps.append(md_step * 2.0 / 1000)  # Assume 2 fs timestep
                    md_step += 1
                except (ValueError, IndexError):
                    pass

            # Look for temperature info (T= in OUTCAR)
            if " T= " in line:
                try:
                    # Format: "T= 1400.0"
                    temp = float(line.split("T=")[1].split()[0])
                    temperatures.append(temp)
                except (ValueError, IndexError):
                    pass

        # Fallback: estimate from frame count if parsing fails
        if not temperatures:
            LOGGER.warning(
                "Could not parse temperatures from OUTCAR, using dummy values"
            )
            temperatures = [1400.0] * len(frames)

        if not energies:
            LOGGER.warning("Could not parse energies from OUTCAR, using frame energies")
            energies = [frame.get_potential_energy() for frame in frames]

        if not times_ps:
            times_ps = [i * 2.0 / 1000 for i in range(len(frames))]

        # Ensure same length
        min_len = min(len(frames), len(temperatures), len(energies), len(times_ps))
        return (
            temperatures[:min_len],
            energies[:min_len],
            times_ps[:min_len],
        )

    def identify_equilibrated_window(
        self, trajectory_data: dict, config: VASPEquilibrationConfig
    ) -> tuple[int, int]:
        """
        Identify equilibrated window in trajectory using sliding window.

        Finds the longest contiguous region post-burn-in where temperature
        stays within tolerance of target (|T - T_target| < tolerance).
        Returns the frame indices of this region.

        Raises:
            ValueError if no stable equilibrated window found.

        Returns:
            (start_frame_idx, end_frame_idx)
        """
        temperatures = np.array(trajectory_data["temperatures"])
        frames = trajectory_data["frames"]

        # Skip burn-in
        burn_in_frames = int(config.burn_in_ps / (config.md_timestep_fs / 1000))
        if burn_in_frames >= len(frames):
            burn_in_frames = len(frames) // 3

        LOGGER.info(f"Discarding first {burn_in_frames} frames as burn-in")

        # Calculate window size
        window_frames = int(config.stability_window_ps / (config.md_timestep_fs / 1000))
        if window_frames > len(frames) - burn_in_frames:
            window_frames = len(frames) - burn_in_frames

        LOGGER.info(
            f"Sliding window size: {window_frames} frames ({config.stability_window_ps} ps)"
        )

        # Slide window and find longest stable region
        best_start = -1
        best_end = -1
        best_duration_frames = 0

        remaining_temps = temperatures[burn_in_frames:]

        for start_offset in range(len(remaining_temps) - window_frames + 1):
            end_offset = start_offset + window_frames
            window_temps = remaining_temps[start_offset:end_offset]
            mean_temp = np.mean(window_temps)

            # Check if this window meets stability criterion
            if (
                abs(mean_temp - config.temperature_target_k)
                < config.temperature_tolerance_k
            ):
                # Track longest stable window
                if window_frames > best_duration_frames:
                    best_start = burn_in_frames + start_offset
                    best_end = burn_in_frames + end_offset
                    best_duration_frames = window_frames

        # If a stable window was found, try to extend it
        if best_start != -1:
            # Extend backward from best_start
            while best_start > burn_in_frames:
                prev_temp = temperatures[best_start - 1]
                if (
                    abs(prev_temp - config.temperature_target_k)
                    < config.temperature_tolerance_k
                ):
                    best_start -= 1
                else:
                    break

            # Extend forward from best_end
            while best_end < len(temperatures):
                next_temp = temperatures[best_end]
                if (
                    abs(next_temp - config.temperature_target_k)
                    < config.temperature_tolerance_k
                ):
                    best_end += 1
                else:
                    break

            best_duration_ps = (best_end - best_start) * config.md_timestep_fs / 1000
            mean_temp_in_window = np.mean(temperatures[best_start:best_end])
            std_temp_in_window = np.std(temperatures[best_start:best_end])

            LOGGER.info(
                f"Found stable equilibrated window: frames {best_start}–{best_end} "
                f"({best_duration_ps:.1f} ps), "
                f"T = {mean_temp_in_window:.1f} ± {std_temp_in_window:.1f} K"
            )

            return best_start, best_end

        # No stable window found
        raise ValueError(
            f"No stable equilibrated window found where |T - {config.temperature_target_k} K| < "
            f"{config.temperature_tolerance_k} K for ≥ {config.stability_window_ps} ps"
        )

    def select_seed_frames(
        self,
        trajectory_data: dict,
        eq_start_idx: int,
        eq_end_idx: int,
        config: AnalysisConfig,
        vasp_config: VASPEquilibrationConfig,
    ) -> list[Atoms]:
        """
        Select decorrelated seed frames from equilibrated window.

        Returns:
            List of Atoms objects (deep copies)
        """
        frames = trajectory_data["frames"]
        times = np.array(trajectory_data["times_ps"])

        # Calculate stride in frames
        stride_frames = int(
            config.seed_spacing_ps / (vasp_config.md_timestep_fs / 1000)
        )
        stride_frames = max(stride_frames, 1)

        LOGGER.info(
            f"Selecting {config.n_seeds} seeds with stride {stride_frames} frames "
            f"({config.seed_spacing_ps} ps)"
        )

        selected_indices = []
        for i in range(config.n_seeds):
            idx = eq_start_idx + i * stride_frames
            if idx >= eq_end_idx:
                LOGGER.warning(
                    f"Cannot fit {config.n_seeds} seeds in equilibrated window "
                    f"(got {len(selected_indices)})"
                )
                break
            selected_indices.append(idx)

        seeds = [frames[idx].copy() for idx in selected_indices]
        LOGGER.info(f"Selected {len(seeds)} seed frames at indices {selected_indices}")

        return seeds
