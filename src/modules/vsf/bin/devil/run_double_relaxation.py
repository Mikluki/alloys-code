#!/usr/bin/env python3
"""Run double relaxation workflow"""

# ============================================================================
# CONFIGURATION - Edit these
# ============================================================================

DIRECTORIES_PATTERN = "*/struct*"
MAX_JOBS = 10
SLEEP_TIME = 30
MAX_DURATION_HOURS = None  # None = no limit

VASP_SETUP = "module load vasp/6.4"
VASP_NTASKS = 128
VASP_NCORE = 32
VASP_KPAR = 4
VASP_ALGO = "Normal"
VASP_PARTITION = None  # or "gpu" etc

STATE_FILE = "vasp-devil-state.json"
VERBOSE = True
DRY_RUN = True

# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    import logging
    import sys
    from datetime import datetime
    from pathlib import Path

    from vsf.bin.devil.core import find_directories
    from vsf.bin.devil.engine import VASPDevil
    from vsf.bin.devil.stages import create_double_relaxation_workflow
    from vsf.logging import setup_logging

    (log_dir := Path("logs")).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = log_dir / f"xx-vasp-devil-{timestamp}.log"
    LOGGER = setup_logging(
        log_file=log_file,
        console_level=logging.DEBUG,
        file_level=logging.DEBUG,
    )

    LOGGER.info(f"Log file saved at '{log_file}'")

    # Find directories
    directories = find_directories(DIRECTORIES_PATTERN)
    if not directories:
        LOGGER.error(f"No directories found: {DIRECTORIES_PATTERN}")
        sys.exit(1)

    LOGGER.info(f"Found {len(directories)} directories")

    # Validate VASP params
    if VASP_NTASKS % (VASP_NCORE * VASP_KPAR) != 0:
        divisor = VASP_NCORE * VASP_KPAR
        LOGGER.error(
            f"Invalid parallelization: ntasks ({VASP_NTASKS}) must be divisible by "
            f"ncore × kpar = {VASP_NCORE} × {VASP_KPAR} = {divisor}"
        )
        sys.exit(1)

    # Create workflow
    slurm_kwargs = {}
    if VASP_PARTITION:
        slurm_kwargs["partition"] = VASP_PARTITION

    workflow = create_double_relaxation_workflow(
        ntasks=VASP_NTASKS,
        ncore=VASP_NCORE,
        kpar=VASP_KPAR,
        algo=VASP_ALGO,
        **slurm_kwargs,
    )

    config = {
        "max_jobs": MAX_JOBS,
        "vasp_setup": VASP_SETUP,
        "max_duration_hours": MAX_DURATION_HOURS,
    }

    devil = VASPDevil(
        workflow=workflow,
        config=config,
        directories=directories,
        state_file=Path(STATE_FILE),
        sleep_time=SLEEP_TIME,
    )

    devil.run(dry_run=DRY_RUN)
