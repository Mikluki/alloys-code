import logging
from dataclasses import dataclass
from typing import Optional

LOGGER = logging.getLogger(__name__)

# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class VASPConfig:
    """Parameters for identifying equilibrated window in VASP MD."""

    burn_in_ps: float
    stability_window_ps: float
    temperature_target_k: float
    temperature_tolerance_k: float
    md_timestep_fs_potim: float = 2.0  # Timestep from VASP POTIM


@dataclass
class GNNMDConfig:
    """Parameters for GNN MD simulation."""

    thermostat_type: str  # "nose_hoover_chain" or "langevin"
    simulation_duration_ps: float
    md_timestep_fs_potim: float  # Timestep in femtoseconds
    target_temperature_k: float
    nose_hoover_damping_fs: Optional[float] = (
        None  # Damping time for Nosé-Hoover (units: fs)
    )
    langevin_friction_coeff: Optional[float] = (
        None  # Friction coefficient for Langevin (units: 1/fs)
    )


@dataclass
class StabilityCheckConfig:
    """Thresholds for detecting collapse."""

    f_max_threshold_ev_a: float
    r_min_threshold_a: float
    energy_spike_threshold_ev: float
    stability_check_interval_steps: int = (
        5  # Full stability checks every N steps (10 fs at 2 fs/step)
    )


@dataclass
class AnalysisConfig:
    """Parameters for trajectory analysis and seeding."""

    n_seeds: int
    seed_spacing_ps: float
    rdf_r_max_a: float
    rdf_dr_a: float
    pre_collapse_ps: float = 2.0  # Capture 2 ps before collapse
    post_collapse_ps: float = 2.0  # Capture 2 ps after collapse detected
    trajectory_stride: float = 5.0


@dataclass
class LoggingConfig:
    """Control logging, profiling, and data caching verbosity."""

    profile_summary_interval_steps: int = 50  # INFO log timing summary every N steps
