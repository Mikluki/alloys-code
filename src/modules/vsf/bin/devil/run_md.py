#!/usr/bin/env python3
"""Run MD orchestration with timer-based graceful shutdown"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these
# ============================================================================

CALC_DIR = "./md-run"  # Path to calculation directory

# TERMINATION CONDITIONS (choose one or more, None = no limit):
MAX_ITERATIONS = None  # Stop after N MD runs (e.g., 10)
MAX_DURATION_HOURS = None  # Stop after N hours (e.g., 24.0)
# Manual stop: Press Ctrl+C to gracefully interrupt and write STOPCAR

VASP_SETUP = "module load vasp/6.4.3_intel"
NODELIST = "node01,node02,node03,node04,node05,node06"
# NODELIST = "ct01,ct02,ct03,ct04,ct06,ct08,ct09,ct10,gn01,gn02,gn03,gn04,gn05,gn06,gn07,gn08,gn09,gn10,gn11,gn12,gn13,gn14,gn15,gn16,gn17,gn18,gn18,gn19,gn20,gn21"
NTASKS = 32
NCORE = 32
KPAR = 1
ALGO = "Normal"

SLEEP_TIME = 60  # Polling interval in seconds

STATE_FILE = "md-state.json"
VERBOSE = True
DRY_RUN = False


# ============================================================================
# EXECUTION
# ============================================================================


def _handle_resume_if_needed(calc_dir: Path, logger):
    """Check for OUTCAR (crash recovery) and optionally backup before resume."""
    outcar = calc_dir / "OUTCAR"

    if not outcar.exists():
        return  # Normal startup, no resume

    logger.warning("OUTCAR found - possible incomplete run or cluster crash")

    response = input("Backup OUTCAR/CONTCAR before resuming? (y/n): ").strip().lower()

    if response == "y":
        from datetime import datetime as dt

        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
        bak_dir = calc_dir / "bak" / f"crash-backup-{timestamp}"
        bak_dir.mkdir(parents=True, exist_ok=True)

        for fname in ["OUTCAR", "CONTCAR"]:
            src = calc_dir / fname
            if src.exists():
                src.rename(bak_dir / fname)

        logger.info(f"✓ Backed up to bak/crash-backup-{timestamp}/")
    else:
        logger.info("Skipping backup, resuming from existing state...")


if __name__ == "__main__":

    from vsf.bin.devil.md_runner import MDConfig, MDOrchestrator
    from vsf.logging import setup_logging

    # Setup logging

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (log_dir := Path("logs")).mkdir(exist_ok=True)
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

    # Handle crash recovery (before creating orchestrator)
    _handle_resume_if_needed(calc_dir, LOGGER)

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
