#!/usr/bin/env python3
"""Run MD orchestration with timer-based graceful shutdown"""

# ============================================================================
# CONFIGURATION - Edit these
# ============================================================================

CALC_DIR = "./md-run"  # Path to calculation directory

# TERMINATION CONDITIONS (choose one or more, None = no limit):
MAX_ITERATIONS = None  # Stop after N MD runs (e.g., 10)
MAX_DURATION_HOURS = None  # Stop after N hours (e.g., 24.0)
# Manual stop: Press Ctrl+C to gracefully interrupt and write STOPCAR

VASP_SETUP = "source /trinity/home/p.zhilyaev/mklk/scripts-run/sif-cpu-ifort.sh"
NODELIST = "ct01,ct02,ct03,ct04,ct06,ct08,ct09,ct10,gn01,gn02,gn03,gn04,gn05,gn06,gn07,gn08,gn09,gn10,gn11,gn12,gn13,gn14,gn15,gn16,gn17,gn18,gn18,gn19,gn20,gn21"
NTASKS = 16
NCORE = 16
KPAR = 1
ALGO = "None"

SLEEP_TIME = 60  # Polling interval in seconds

STATE_FILE = "md-state.json"
VERBOSE = True
DRY_RUN = False

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    import logging
    import sys
    from datetime import datetime
    from pathlib import Path

    from vsf.bin.devil.md_runner import MDConfig, MDOrchestrator
    from vsf.logging import setup_logging

    # Setup logging

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"md_orchestrator_{timestamp}.log"

    (log_dir := Path("logs")).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = log_dir / f"xx-vasp-devil-md-{timestamp}.log"
    LOGGER = setup_logging(
        log_file=log_file,
        console_level=logging.DEBUG,
        file_level=logging.DEBUG,
    )

    LOGGER.info(f"Log file saved at '{log_file}'")

    # Validate directory
    calc_dir = Path(CALC_DIR)
    if not calc_dir.exists():
        LOGGER.error(f"Calculation directory not found: {calc_dir}")
        sys.exit(1)

    # Check for required input files
    required_files = ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]
    missing = [f for f in required_files if not (calc_dir / f).exists()]
    if missing:
        LOGGER.error(f"Missing required files: {missing}")
        sys.exit(1)

    LOGGER.info(f"✓ Calculation directory: {calc_dir}")

    # Create config
    config = MDConfig(
        vasp_setup=VASP_SETUP,
        ntasks=NTASKS,
        ncore=NCORE,
        kpar=KPAR,
        algo=ALGO,
        nodelist=NODELIST,
    )

    # Create orchestrator
    orchestrator = MDOrchestrator(
        calc_dir=calc_dir,
        config=config,
        state_file=Path(STATE_FILE),
        max_iterations=MAX_ITERATIONS,
        max_duration_hours=MAX_DURATION_HOURS,
        sleep_time=SLEEP_TIME,
    )

    # Run
    orchestrator.run(dry_run=DRY_RUN)
