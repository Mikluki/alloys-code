import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import BaseCalculator
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from .md_config import (
    AnalysisConfig,
    GNNMDConfig,
    LoggingConfig,
    StabilityCheckConfig,
)

LOGGER = logging.getLogger(__name__)


# ============================================================================
# GNNMDRunner
# ============================================================================


class GNNMDRunner:
    """Execute GNN MD simulation with stability monitoring."""

    def __init__(self):
        pass

    def run_simulation(
        self,
        seed_atoms: Atoms,
        calculator: BaseCalculator,
        output_dir: Path,
        md_config: GNNMDConfig,
        stability_check_config: StabilityCheckConfig,
        analysis_config: AnalysisConfig,
        logging_config: LoggingConfig,
    ) -> dict:
        """
        Run MD simulation from seed atoms.

        Saves trajectory to output_dir/trajectory.json, logs metadata to
        output_dir/metadata.json.

        Args:
            seed_atoms: Starting structure (Atoms)
            calculator: Instantiated calculator (BaseNN or similar)
            output_dir: Directory to write results
            md_config: MD parameters
            stability_check_config: Collapse detection thresholds
            analysis_config: Analysis parameters (pre/post-collapse frame capture, stride)
            logging_config: Logging and profiling control

        Returns:
            {
                "n_steps": int,
                "stable_until_ps": float | None,
                "collapse_time_ps": float | None,
                "collapse_reason": str | None,
            }
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        atoms = seed_atoms.copy()
        atoms.calc = calculator

        # Initialize velocities for consistent initial conditions
        # Ensures reproducible state; thermostat maintains T, not establishes it
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=md_config.target_temperature_k
        )

        # Set up thermostat
        dt = md_config.md_timestep_fs * units.fs
        T = md_config.target_temperature_k

        if md_config.thermostat_type == "nose_hoover_chain":
            if md_config.nose_hoover_damping_fs is None:
                raise ValueError(
                    "nose_hoover_damping_fs required for nose_hoover_chain"
                )
            tdamp = md_config.nose_hoover_damping_fs * units.fs
            dyn = NoseHooverChainNVT(
                atoms,
                timestep=dt,
                temperature_K=T,
                tdamp=tdamp,
            )
            LOGGER.info(
                f"Initialized NoseHooverChainNVT: T={T} K, dt={md_config.md_timestep_fs} fs, "
                f"damping={md_config.nose_hoover_damping_fs} fs"
            )
        elif md_config.thermostat_type == "langevin":
            if md_config.langevin_friction_coeff is None:
                raise ValueError("langevin_friction_coeff required for langevin")
            dyn = Langevin(
                atoms,
                timestep=dt,
                temperature_K=T,
                friction=md_config.langevin_friction_coeff / units.fs,
            )
            LOGGER.info(
                f"Initialized Langevin: T={T} K, dt={md_config.md_timestep_fs} fs, "
                f"friction={md_config.langevin_friction_coeff} fs^-1"
            )
        else:
            raise ValueError(f"Unknown thermostat_type: {md_config.thermostat_type}")

        # Monitoring arrays
        stability_monitor = StabilityMonitor(stability_check_config)
        trajectory = []
        temperatures = []
        energies = []
        max_forces = []  # Max force per step (always saved)
        forces_strided = []  # Full force arrays at stride intervals
        stress_strided = []  # Stress tensors at stride intervals
        times_ps = []
        times_strided_ps = []  # Times corresponding to strided saves

        # Timing buffers (rolling window of last 50 measurements)
        timings = {
            "run": deque(maxlen=50),
            "energy": deque(maxlen=50),
            "forces": deque(maxlen=50),
            "stress": deque(maxlen=50),
            "kin_temp": deque(maxlen=50),
            "stability": deque(maxlen=50),
            "total": deque(maxlen=50),
        }

        # Calculate pre/post-collapse frame counts from AnalysisConfig
        pre_collapse_steps = int(
            analysis_config.pre_collapse_ps / (md_config.md_timestep_fs / 1000)
        )
        post_collapse_steps = int(
            analysis_config.post_collapse_ps / (md_config.md_timestep_fs / 1000)
        )

        n_total_steps = int(
            md_config.simulation_duration_ps / (md_config.md_timestep_fs / 1000)
        )
        collapse_time_ps: Optional[float] = None
        collapse_reason: Optional[str] = None
        stable_until_ps: Optional[float] = None
        collapse_step: Optional[int] = None

        trajectory_stride = analysis_config.trajectory_stride

        LOGGER.info(
            f"Starting MD: {md_config.simulation_duration_ps} ps ({n_total_steps} steps), "
            f"trajectory stride: {trajectory_stride}"
        )

        # Main MD loop
        for step in range(n_total_steps):
            step_t0 = time.perf_counter()

            # MD step
            try:
                t_run_start = time.perf_counter()
                dyn.run(1)
                timings["run"].append(time.perf_counter() - t_run_start)
            except Exception as e:
                LOGGER.error(f"MD step {step} failed with exception: {e}")
                collapse_time_ps = step * md_config.md_timestep_fs / 1000
                collapse_reason = f"Exception: {e}"
                break

            # Energy
            t_energy_start = time.perf_counter()
            energy = atoms.get_potential_energy()
            timings["energy"].append(time.perf_counter() - t_energy_start)

            # Forces
            t_forces_start = time.perf_counter()
            forces = atoms.get_forces()
            timings["forces"].append(time.perf_counter() - t_forces_start)

            # Kinetic energy and temperature
            t_kin_start = time.perf_counter()
            kinetic_energy = atoms.get_kinetic_energy()
            temperature = 2 * kinetic_energy / (3 * len(atoms) * units.kB)
            timings["kin_temp"].append(time.perf_counter() - t_kin_start)

            # Store trajectory (all frames)
            current_time_ps = step * md_config.md_timestep_fs / 1000
            trajectory.append(atoms.copy())
            energies.append(energy)
            temperatures.append(temperature)
            times_ps.append(current_time_ps)

            # Max force (always save)
            f_max = np.max(np.abs(forces))
            max_forces.append(f_max)

            # Strided save: forces and stress (expensive operations)
            if step % trajectory_stride == 0:
                # Save full force array
                forces_strided.append(forces.tolist())

                # Compute and save stress tensor
                t_stress_start = time.perf_counter()
                stress = atoms.get_stress()
                timings["stress"].append(time.perf_counter() - t_stress_start)
                stress_strided.append(stress.tolist())

                times_strided_ps.append(current_time_ps)

            # Fast check: max force every step
            # Full stability checks: periodic
            t_stab_start = time.perf_counter()
            stable = True
            reason = None

            if step % stability_check_config.stability_check_interval_steps == 0:
                # Do full checks periodically
                stable, reason = stability_monitor.check_frame(atoms, forces, energy)
            else:
                # Only check max force on other steps
                if f_max > stability_check_config.f_max_threshold_ev_a:
                    stable = False
                    reason = f"Max force {f_max:.3f} eV/Å exceeds threshold {stability_check_config.f_max_threshold_ev_a:.3f}"

            timings["stability"].append(time.perf_counter() - t_stab_start)

            # Total step time
            timings["total"].append(time.perf_counter() - step_t0)

            # INFO: periodic summary
            if (step + 1) % logging_config.profile_summary_interval_steps == 0:
                avg_times = {k: np.mean(list(v)) * 1000 for k, v in timings.items()}
                remaining_steps = n_total_steps - step - 1
                est_total_s = (remaining_steps * avg_times["total"]) / 1000
                LOGGER.info(
                    f"Steps {step + 1:6d} / {n_total_steps} | "
                    f"Avg step time: {avg_times['total']:.2f}ms | "
                    f"Breakdown: run={avg_times['run']:.2f}ms "
                    f"energy={avg_times['energy']:.2f}ms "
                    f"forces={avg_times['forces']:.2f}ms "
                    f"kin_temp={avg_times['kin_temp']:.2f}ms "
                    f"stab={avg_times['stability']:.2f}ms | "
                    f"Est. remaining: {est_total_s:.1f}s"
                )

            # Check stability and handle collapse
            if not stable:
                LOGGER.warning(
                    f"Collapse detected at {current_time_ps:.2f} ps: {reason}"
                )
                collapse_time_ps = current_time_ps
                collapse_reason = reason
                stable_until_ps = times_ps[-1] if times_ps else None
                collapse_step = step

                # Continue post_collapse_steps more steps to capture post-collapse dynamics
                for extra_step in range(post_collapse_steps):
                    try:
                        dyn.run(1)
                        post_step = step + 1 + extra_step
                        post_time = post_step * md_config.md_timestep_fs / 1000

                        # Store post-collapse frames
                        trajectory.append(atoms.copy())
                        energies.append(atoms.get_potential_energy())
                        temperatures.append(
                            2 * atoms.get_kinetic_energy() / (3 * len(atoms) * units.kB)
                        )
                        times_ps.append(post_time)

                        # Compute and append max force
                        post_forces = atoms.get_forces()
                        post_f_max = np.max(np.abs(post_forces))
                        max_forces.append(post_f_max)

                    except Exception as e:
                        LOGGER.debug(
                            f"Failed to run post-collapse step {extra_step}: {e}"
                        )
                        break

                # Capture pre-collapse frames (previous pre_collapse_ps)
                pre_start_idx = max(0, collapse_step - pre_collapse_steps)
                pre_frames = trajectory[pre_start_idx : collapse_step + 1]
                pre_times = times_ps[pre_start_idx : collapse_step + 1]

                # Capture post-collapse frames (from collapse onwards)
                post_frames = trajectory[collapse_step + 1 :]
                post_times = times_ps[collapse_step + 1 :]

                # Save trajectory data
                traj_data = {
                    "trajectory": [self._atoms_to_dict(frame) for frame in trajectory],
                    "trajectory_times_ps": times_ps,
                    "trajectory_temperatures": temperatures,
                    "trajectory_energies": energies,
                    "trajectory_max_forces": max_forces,
                    "trajectory_forces_strided": forces_strided,
                    "trajectory_stress_strided": stress_strided,
                    "trajectory_times_strided_ps": times_strided_ps,
                    "trajectory_stride": trajectory_stride,
                    "collapse_step": collapse_step,
                    "collapse_time_ps": collapse_time_ps,
                    "collapse_reason": collapse_reason,
                    "pre_collapse_frames": [
                        self._atoms_to_dict(frame) for frame in pre_frames
                    ],
                    "pre_collapse_times_ps": pre_times,
                    "post_collapse_frames": [
                        self._atoms_to_dict(frame) for frame in post_frames
                    ],
                    "post_collapse_times_ps": post_times,
                }

                (output_dir / "trajectory.json").write_text(json.dumps(traj_data))
                LOGGER.info(f"Saved trajectory to {output_dir / 'trajectory.json'}")

                # Save metadata
                metadata = {
                    "n_steps": step,
                    "stable_until_ps": stable_until_ps,
                    "collapse_time_ps": collapse_time_ps,
                    "collapse_reason": collapse_reason,
                    "collapse_step": collapse_step,
                    "pre_collapse_frames_count": len(pre_frames),
                    "post_collapse_frames_count": len(post_frames),
                    "md_config": self._config_to_dict(md_config),
                }
                (output_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2)
                )

                return metadata

        # Completed without collapse
        traj_data = {
            "trajectory": [self._atoms_to_dict(frame) for frame in trajectory],
            "trajectory_times_ps": times_ps,
            "trajectory_temperatures": temperatures,
            "trajectory_energies": energies,
            "trajectory_max_forces": max_forces,
            "trajectory_forces_strided": forces_strided,
            "trajectory_stress_strided": stress_strided,
            "trajectory_times_strided_ps": times_strided_ps,
            "trajectory_stride": trajectory_stride,
            "collapse_step": None,
            "collapse_time_ps": None,
            "collapse_reason": None,
            "pre_collapse_frames": [],
            "pre_collapse_times_ps": [],
            "post_collapse_frames": [],
            "post_collapse_times_ps": [],
        }

        (output_dir / "trajectory.json").write_text(json.dumps(traj_data))
        LOGGER.info(f"Saved complete trajectory to {output_dir / 'trajectory.json'}")

        metadata = {
            "n_steps": n_total_steps,
            "stable_until_ps": None,
            "collapse_time_ps": None,
            "collapse_reason": None,
            "collapse_step": None,
            "pre_collapse_frames_count": 0,
            "post_collapse_frames_count": 0,
            "md_config": self._config_to_dict(md_config),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        LOGGER.info(
            f"Simulation completed without collapse after {n_total_steps} steps"
        )

        return metadata

    @staticmethod
    def _atoms_to_dict(atoms: Atoms) -> dict:
        """Convert Atoms to serializable dict."""
        return {
            "positions": atoms.positions.tolist(),
            "symbols": atoms.get_chemical_symbols(),
            "cell": atoms.cell.cellpar().tolist(),
            "pbc": atoms.pbc.tolist(),
        }

    @staticmethod
    def _config_to_dict(config: GNNMDConfig) -> dict:
        """Convert MDConfig to dict."""
        return {
            "thermostat_type": config.thermostat_type,
            "simulation_duration_ps": config.simulation_duration_ps,
            "md_timestep_fs": config.md_timestep_fs,
            "target_temperature_k": config.target_temperature_k,
            "nose_hoover_damping_fs": config.nose_hoover_damping_fs,
            "langevin_friction_coeff": config.langevin_friction_coeff,
        }


# ============================================================================
# StabilityMonitor
# ============================================================================


class StabilityMonitor:
    """Monitor for collapse detection based on force, distance, and energy thresholds."""

    def __init__(self, config: StabilityCheckConfig):
        self.config = config
        self.last_energy: Optional[float] = None

    def check_frame(
        self,
        atoms: Atoms,
        forces: np.ndarray,
        energy: float,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if frame is stable.

        Args:
            atoms: Atoms object
            forces: Forces array (N, 3)
            energy: Potential energy (scalar)

        Returns:
            (is_stable, reason_if_unstable)
        """
        f_max = np.max(np.abs(forces))

        # Check max force threshold
        if f_max > self.config.f_max_threshold_ev_a:
            return (
                False,
                f"Max force {f_max:.3f} eV/Å exceeds threshold {self.config.f_max_threshold_ev_a:.3f}",
            )

        # Check minimum interatomic distance
        if len(atoms) > 1:
            distances = atoms.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            r_min = np.min(distances)
            if r_min < self.config.r_min_threshold_a:
                return (
                    False,
                    f"Min distance {r_min:.3f} Å below threshold {self.config.r_min_threshold_a:.3f}",
                )

        # Energy spike: only collapse if BOTH spike is large AND forces are moderately elevated
        # This avoids false positives from thermal noise in liquids
        if self.last_energy is not None:
            e_spike = abs(energy - self.last_energy)
            if e_spike > self.config.energy_spike_threshold_ev:
                # Only flag as collapse if forces are also elevated (50% of collapse threshold)
                f_threshold_combined = self.config.f_max_threshold_ev_a * 0.5
                if f_max > f_threshold_combined:
                    return (
                        False,
                        f"Energy spike {e_spike:.3f} eV + elevated forces {f_max:.3f} eV/Å",
                    )
                else:
                    # Log the spike even though it didn't trigger collapse
                    LOGGER.warning(
                        f"Energy spike {e_spike:.3f} eV (threshold {self.config.energy_spike_threshold_ev:.3f}) "
                        f"detected but forces {f_max:.3f} eV/Å below combined threshold {f_threshold_combined:.3f} — ignoring"
                    )

        self.last_energy = energy
        return True, None
