"""
MD simulation and analysis module for GNN validation against VASP.

Combines VASP trajectory parsing, GNN MD execution with stability monitoring,
and ensemble property analysis (RDF, MSD, thermodynamic metrics).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from .md_config import AnalysisConfig

LOGGER = logging.getLogger(__name__)


# ============================================================================
# EnsembleAnalyzer
# ============================================================================


class EnsembleAnalyzer:
    """Compute ensemble properties from trajectories."""

    def __init__(self):
        pass

    def load_trajectory_from_json(self, traj_path: Path) -> dict:
        """
        Load trajectory from JSON file.

        Returns:
            {
                "frames": list[Atoms],
                "times_ps": list[float],
                "temperatures": list[float],
                "energies": list[float],
                "collapse_info": dict | None,
            }
        """
        with open(traj_path) as f:
            data = json.load(f)

        frames = [self._dict_to_atoms(d) for d in data.get("trajectory", [])]
        times_ps = data.get("trajectory_times_ps", [])
        temperatures = data.get("trajectory_temperatures", [])
        energies = data.get("trajectory_energies", [])

        collapse_info = None
        if data.get("collapse_time_ps") is not None:
            collapse_info = {
                "collapse_time_ps": data["collapse_time_ps"],
                "collapse_reason": data["collapse_reason"],
                "pre_collapse_frames": [
                    self._dict_to_atoms(d) for d in data.get("pre_collapse_frames", [])
                ],
                "post_collapse_frames": [
                    self._dict_to_atoms(d) for d in data.get("post_collapse_frames", [])
                ],
            }

        return {
            "frames": frames,
            "times_ps": times_ps,
            "temperatures": temperatures,
            "energies": energies,
            "collapse_info": collapse_info,
        }

    def compute_temperature_statistics(
        self,
        trajectory: dict,
        window_start_ps: float = 0,
        window_end_ps: Optional[float] = None,
    ) -> dict:
        """
        Compute temperature statistics over a time window.

        Returns:
            {
                "mean_k": float,
                "std_k": float,
                "min_k": float,
                "max_k": float,
                "n_frames": int,
            }
        """
        times = np.array(trajectory["times_ps"])
        temps = np.array(trajectory["temperatures"])

        if window_end_ps is None:
            window_end_ps = times[-1] if len(times) > 0 else 0

        mask = (times >= window_start_ps) & (times <= window_end_ps)
        window_temps = temps[mask]

        if len(window_temps) == 0:
            LOGGER.warning(
                f"No frames in window [{window_start_ps}, {window_end_ps}] ps"
            )
            return {
                "mean_k": None,
                "std_k": None,
                "min_k": None,
                "max_k": None,
                "n_frames": 0,
            }

        return {
            "mean_k": float(np.mean(window_temps)),
            "std_k": float(np.std(window_temps)),
            "min_k": float(np.min(window_temps)),
            "max_k": float(np.max(window_temps)),
            "n_frames": len(window_temps),
        }

    def compute_energy_statistics(
        self,
        trajectory: dict,
        window_start_ps: float = 0,
        window_end_ps: Optional[float] = None,
    ) -> dict:
        """
        Compute energy statistics over a time window.

        Returns:
            {
                "mean_ev": float,
                "std_ev": float,
                "min_ev": float,
                "max_ev": float,
                "n_frames": int,
            }
        """
        times = np.array(trajectory["times_ps"])
        energies = np.array(trajectory["energies"])

        if window_end_ps is None:
            window_end_ps = times[-1] if len(times) > 0 else 0

        mask = (times >= window_start_ps) & (times <= window_end_ps)
        window_energies = energies[mask]

        if len(window_energies) == 0:
            LOGGER.warning(
                f"No frames in window [{window_start_ps}, {window_end_ps}] ps"
            )
            return {
                "mean_ev": None,
                "std_ev": None,
                "min_ev": None,
                "max_ev": None,
                "n_frames": 0,
            }

        return {
            "mean_ev": float(np.mean(window_energies)),
            "std_ev": float(np.std(window_energies)),
            "min_ev": float(np.min(window_energies)),
            "max_ev": float(np.max(window_energies)),
            "n_frames": len(window_energies),
        }

    def compute_rdf(
        self,
        trajectory: dict,
        pair_types: list[tuple[str, str]],
        r_max: float,
        dr: float,
        window_start_ps: float = 0,
        window_end_ps: Optional[float] = None,
    ) -> dict:
        """
        Compute radial distribution functions using freud.

        Provides properly normalized g(r) accounting for volume, density, and periodicity.

        Args:
            trajectory: Trajectory dict
            pair_types: List of (symbol1, symbol2) tuples, e.g. [("Al", "Al"), ("Al", "Cu")]
            r_max: Maximum radius (Å)
            dr: Bin width (Å)
            window_start_ps: Start of analysis window
            window_end_ps: End of analysis window

        Returns:
            {
                pair_key: {
                    "r": list[float],
                    "g_r": list[float],
                    "first_peak_r": float | None,
                    "first_peak_g": float | None,
                    "coordination_number": float,
                }
            }
        """
        # Try to import freud; fail gracefully if unavailable
        try:
            import freud
        except ImportError:
            LOGGER.error(
                "freud not installed; RDF computation unavailable. "
                "Install with: pip install freud-analysis"
            )
            return {}

        times = np.array(trajectory["times_ps"])
        frames = trajectory["frames"]

        if window_end_ps is None:
            window_end_ps = times[-1] if len(times) > 0 else 0

        mask = (times >= window_start_ps) & (times <= window_end_ps)
        window_frames = [frames[i] for i, m in enumerate(mask) if m]

        if not window_frames:
            LOGGER.warning(
                f"No frames in RDF window [{window_start_ps}, {window_end_ps}] ps"
            )
            return {}

        LOGGER.info(f"Computing RDF from {len(window_frames)} frames using freud")

        rdfs = {}

        for sym1, sym2 in pair_types:
            pair_key = f"{sym1}-{sym2}"

            # Initialize RDF computer
            n_bins = int(r_max / dr)
            rdf_computer = freud.density.RDF(bins=n_bins, r_max=r_max)

            # Compute RDF over window frames
            for atoms in window_frames:
                box = freud.box.Box.from_matrix(atoms.cell.array)
                positions = atoms.get_positions()
                symbols = atoms.get_chemical_symbols()

                # Get atom indices for this pair type
                indices_1 = np.array([i for i, s in enumerate(symbols) if s == sym1])
                indices_2 = np.array([i for i, s in enumerate(symbols) if s == sym2])

                if len(indices_1) == 0 or len(indices_2) == 0:
                    continue

                # Extract subset positions
                pos_1 = positions[indices_1]
                pos_2 = positions[indices_2]

                # Compute RDF
                if sym1 == sym2:
                    # Self-RDF: compute on the subset of atoms
                    rdf_computer.compute((box, pos_1))
                else:
                    # Cross-RDF: query pos_1 against pos_2
                    rdf_computer.compute((box, pos_2), pos_1)

            # Extract normalized g(r) and r values
            g_r = rdf_computer.rdf
            r = rdf_computer.bin_centers

            # Find first peak
            first_peak_idx = None
            first_peak_g = None
            first_peak_r = None
            coord_number = 0.0

            if len(g_r) > 0 and np.max(g_r) > 0:
                first_peak_idx = np.argmax(g_r)
                first_peak_g = float(g_r[first_peak_idx])
                first_peak_r = float(r[first_peak_idx])

                # Coordination number: area under first peak (up to 1.2 × peak location)
                if first_peak_idx is not None:
                    peak_cutoff = int(first_peak_idx * 1.2)
                    coord_number = float(np.sum(g_r[:peak_cutoff]) * dr)

            rdfs[pair_key] = {
                "r": r.tolist(),
                "g_r": g_r.tolist(),
                "first_peak_r": first_peak_r,
                "first_peak_g": first_peak_g,
                "coordination_number": coord_number,
            }

        return rdfs

    def compute_msd(
        self,
        trajectory: dict,
        window_start_ps: float = 0,
        window_end_ps: Optional[float] = None,
    ) -> dict:
        """
        Compute mean squared displacement and diffusion coefficients.

        Returns:
            {
                "msd_times_ps": list[float],
                "msd": list[float],
                "diffusion_coefficient_a2_ps": float | None,
            }
        """
        times = np.array(trajectory["times_ps"])
        frames = trajectory["frames"]

        if window_end_ps is None:
            window_end_ps = times[-1] if len(times) > 0 else 0

        # Find window
        mask = (times >= window_start_ps) & (times <= window_end_ps)
        window_frames = [frames[i] for i, m in enumerate(mask) if m]
        window_times = times[mask]

        if len(window_frames) < 2:
            LOGGER.warning(
                f"Not enough frames for MSD in window [{window_start_ps}, {window_end_ps}] ps"
            )
            return {
                "msd_times_ps": [],
                "msd": [],
                "diffusion_coefficient_a2_ps": None,
            }

        LOGGER.info(f"Computing MSD from {len(window_frames)} frames")

        # Compute MSD relative to first frame in window
        ref_positions = window_frames[0].positions
        msd_array = np.zeros(len(window_frames))

        for i, atoms in enumerate(window_frames):
            displacements = atoms.positions - ref_positions
            msd_array[i] = np.mean(np.sum(displacements**2, axis=1))

        # Fit diffusion coefficient (MSD = 6*D*t)
        if len(window_times) > 2:
            # Use linear fit on later timepoints (avoid ballistic regime)
            t_fit = window_times[len(window_times) // 2 :]
            msd_fit = msd_array[len(window_times) // 2 :]
            if len(t_fit) > 1 and np.max(t_fit) > 0:
                coeffs = np.polyfit(t_fit, msd_fit, 1)
                d_coeff = coeffs[0] / 6.0  # MSD = 6*D*t
            else:
                d_coeff = None
        else:
            d_coeff = None

        return {
            "msd_times_ps": window_times.tolist(),
            "msd": msd_array.tolist(),
            "diffusion_coefficient_a2_ps": d_coeff,
        }

    @staticmethod
    def _dict_to_atoms(data: dict) -> Atoms:
        """Reconstruct Atoms from dict."""
        atoms = Atoms(
            symbols=data["symbols"],
            positions=data["positions"],
            pbc=data["pbc"],
        )
        atoms.set_cell(data["cell"], scale_atoms=False)
        return atoms


# ============================================================================
# TrajectorySet
# ============================================================================


class TrajectorySet:
    """Container for VASP reference + GNN simulation trajectories."""

    def __init__(self):
        self.vasp_trajectory: Optional[dict] = None
        self.gnn_trajectories: dict[str, dict] = {}
        self.metadata: dict = {}

    def set_vasp_reference(self, trajectory: dict, metadata: dict):
        """Set VASP reference trajectory."""
        self.vasp_trajectory = trajectory
        self.metadata["vasp"] = metadata
        LOGGER.info(f"Set VASP reference with {len(trajectory['frames'])} frames")

    def add_gnn_trajectory(self, seed_id: str, trajectory: dict, metadata: dict):
        """Add a GNN simulation trajectory."""
        self.gnn_trajectories[seed_id] = trajectory
        self.metadata[f"gnn_{seed_id}"] = metadata
        LOGGER.info(f"Added GNN trajectory for seed {seed_id}")

    def get_healthy_window(
        self, trajectory: dict, max_time_ps: Optional[float] = None
    ) -> tuple[float, float]:
        """
        Get analysis window for healthy (non-collapsed) trajectory.

        Returns:
            (window_start_ps, window_end_ps)
        """
        if max_time_ps is None:
            times = np.array(trajectory["times_ps"])
            max_time_ps = times[-1] if len(times) > 0 else 0

        # For VASP, use last 20 ps (or full trajectory if shorter)
        window_duration_ps = 20
        assert max_time_ps is not None
        window_start_ps = max(0, max_time_ps - window_duration_ps)

        return window_start_ps, max_time_ps

    def compare_ensembles(
        self, analyzer: EnsembleAnalyzer, config: AnalysisConfig
    ) -> dict:
        """
        Compare VASP and GNN ensemble properties.

        Returns:
            {
                "temperature": {
                    "vasp": dict,
                    gnn_seeds: dict,
                },
                "energy": {...},
                "rdf": {...},
                ...
            }
        """
        if self.vasp_trajectory is None:
            raise ValueError("VASP reference not set")

        # Get healthy windows
        vasp_start, vasp_end = self.get_healthy_window(self.vasp_trajectory)

        results = {
            "temperature": {},
            "energy": {},
            "rdf": {},
            "msd": {},
        }

        # VASP metrics
        LOGGER.info(
            f"Computing VASP metrics for window [{vasp_start:.1f}, {vasp_end:.1f}] ps"
        )
        results["temperature"]["vasp"] = analyzer.compute_temperature_statistics(
            self.vasp_trajectory, vasp_start, vasp_end
        )
        results["energy"]["vasp"] = analyzer.compute_energy_statistics(
            self.vasp_trajectory, vasp_start, vasp_end
        )

        # Common pair types (adjust based on your system)
        pair_types = [
            ("Al", "Al"),
            ("Al", "Cu"),
            ("Al", "Ni"),
            ("Cu", "Cu"),
            ("Cu", "Ni"),
            ("Ni", "Ni"),
        ]
        vasp_rdf = analyzer.compute_rdf(
            self.vasp_trajectory,
            pair_types,
            config.rdf_r_max_a,
            config.rdf_dr_a,
            vasp_start,
            vasp_end,
        )
        results["rdf"]["vasp"] = vasp_rdf

        vasp_msd = analyzer.compute_msd(self.vasp_trajectory, vasp_start, vasp_end)
        results["msd"]["vasp"] = vasp_msd

        # GNN metrics
        results["temperature"]["gnn"] = {}
        results["energy"]["gnn"] = {}
        results["rdf"]["gnn"] = {}
        results["msd"]["gnn"] = {}

        for seed_id, traj in self.gnn_trajectories.items():
            LOGGER.info(f"Computing metrics for GNN seed {seed_id}")

            # For GNN, use all available healthy frames
            gnn_times = np.array(traj["times_ps"])
            if traj["collapse_info"]:
                gnn_end = traj["collapse_info"]["collapse_time_ps"]
                gnn_start = max(0, gnn_end - 5)  # 5 ps before collapse
            else:
                gnn_end = gnn_times[-1] if len(gnn_times) > 0 else 0
                gnn_start = max(0, gnn_end - 20)  # Last 20 ps

            results["temperature"]["gnn"][seed_id] = (
                analyzer.compute_temperature_statistics(traj, gnn_start, gnn_end)
            )
            results["energy"]["gnn"][seed_id] = analyzer.compute_energy_statistics(
                traj, gnn_start, gnn_end
            )

            gnn_rdf = analyzer.compute_rdf(
                traj,
                pair_types,
                config.rdf_r_max_a,
                config.rdf_dr_a,
                gnn_start,
                gnn_end,
            )
            results["rdf"]["gnn"][seed_id] = gnn_rdf

            gnn_msd = analyzer.compute_msd(traj, gnn_start, gnn_end)
            results["msd"]["gnn"][seed_id] = gnn_msd

        return results
