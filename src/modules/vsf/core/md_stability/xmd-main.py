"""
MD simulation and analysis for GNN validation against VASP.

Combines VASP trajectory parsing, GNN MD execution with stability monitoring,
and ensemble property analysis (RDF, MSD, thermodynamic metrics).
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from vsf.core.md_stability.md_analysis import (
    EnsembleAnalyzer,
    TrajectorySet,
)
from vsf.core.md_stability.md_config import (
    AnalysisConfig,
    GNNMDConfig,
    LoggingConfig,
    StabilityCheckConfig,
    VASPConfig,
)
from vsf.core.md_stability.md_runner import GNNMDRunner
from vsf.core.md_stability.md_vasp import VASPTrajectoryAnalyzer
from vsf.energy.energy_source import EnergySource
from vsf.logging import setup_logging

# Parse energy source argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--energy-source", required=True, help="Energy source name from YAML config"
)
args = parser.parse_args()
energy_source = args.energy_source

# Logging
timestamp = datetime.now().strftime("%m-%dT%H:%M:%S")
(log_dir := Path("logs")).mkdir(exist_ok=True)
LOGGER = setup_logging(
    log_file=log_dir / f"x-gnn-md-{energy_source}-{timestamp}.log",
    console_level=logging.DEBUG,
    file_level=logging.DEBUG,
)

# Paths
vasp_dir = Path("AlCuNi_L1915_1400-lght")

# 0. CONFIGS
logging_config = LoggingConfig(
    profile_summary_interval_steps=50,
)

eq_config = VASPConfig(
    burn_in_ps=5,
    stability_window_ps=1,
    temperature_target_k=1400,
    temperature_tolerance_k=50,
    md_timestep_fs_potim=2.0,  # VASP timestep (POTIM)
)

analysis_config = AnalysisConfig(
    n_seeds=5,
    seed_spacing_ps=5,
    rdf_r_max_a=9,
    rdf_dr_a=0.1,
    pre_collapse_ps=0.1,  # 100 fs pre-collapse capture
    post_collapse_ps=0.1,  # 100 fs post-collapse capture
    trajectory_stride=5.0,
)

md_config = GNNMDConfig(
    thermostat_type="nose_hoover_chain",
    simulation_duration_ps=10,  # Collapse timescale: ~1-10 ps
    md_timestep_fs_potim=2.0,
    target_temperature_k=1400,
    nose_hoover_damping_fs=200.0,
)

stability_config = StabilityCheckConfig(
    f_max_threshold_ev_a=100,
    r_min_threshold_a=1.5,
    energy_spike_threshold_ev=15.0,
    stability_check_interval_steps=5,
)

# 1. Parse VASP
LOGGER.info("=" * 80)
LOGGER.info("STEP 1: Load and analyze VASP reference trajectory")
LOGGER.info("=" * 80)

vasp_analyzer = VASPTrajectoryAnalyzer()
# vasp_data = vasp_analyzer.load_trajectory(vasp_dir)
vasp_data = vasp_analyzer.load_trajectories_from_files(
    vasp_dir, md_timestep_fs=md_config.md_timestep_fs_potim
)
LOGGER.info(f"Loaded VASP trajectory: {len(vasp_data['frames'])} frames")

eq_start, eq_end = vasp_analyzer.identify_equilibrated_window(vasp_data, eq_config)
LOGGER.info(f"Equilibrated window: frames {eq_start}–{eq_end}")

seeds = vasp_analyzer.select_seed_frames(
    vasp_data, eq_start, eq_end, analysis_config, eq_config
)
LOGGER.info(f"Selected {len(seeds)} seed frames")

# 2. Run GNN MD for each seed
LOGGER.info("=" * 80)
LOGGER.info("STEP 2: Run GNN MD simulations from seed frames")
LOGGER.info("=" * 80)

runner = GNNMDRunner()
calc = EnergySource[energy_source].get_calculator(device="cuda")

output_base = Path(f"results-{energy_source}-{timestamp}")
output_base.mkdir(exist_ok=True)

for i, seed in enumerate(seeds):
    output_dir = output_base / f"seed-{i}"
    LOGGER.info(f"Running GNN MD for seed {i}...")

    metadata = runner.run_simulation(
        seed,
        calc.ase_calculator,
        output_dir,
        md_config,
        stability_config,
        analysis_config,
        logging_config,
    )

    LOGGER.info(f"Seed {i} result: {metadata}")

# 3. Load and compare ensembles
LOGGER.info("=" * 80)
LOGGER.info("STEP 3: Compare VASP and GNN ensembles")
LOGGER.info("=" * 80)

traj_set = TrajectorySet()
traj_set.set_vasp_reference(vasp_data, {"source": "VASP"})

analyzer = EnsembleAnalyzer()
for i in range(len(seeds)):
    output_dir = output_base / f"seed-{i}"
    gnn_traj = analyzer.load_trajectory_from_json(output_dir / "trajectory.json")
    traj_set.add_gnn_trajectory(
        f"seed_{i}", gnn_traj, {"source": f"GNN-{energy_source}"}
    )

comparison = traj_set.compare_ensembles(analyzer, analysis_config)

LOGGER.info("=" * 80)
LOGGER.info("Ensemble comparison complete")
LOGGER.info("=" * 80)
# LOGGER.info(f"Results: {comparison}")
