#!/usr/bin/env python3
"""
GNN Runner - Subprocess executable for batch GNN energy calculations.

Entry point for: python -m gnn_runner
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from ..energy.energy_source import EnergySource
from ..logging import setup_logging
from .factory import create_calculator
from .interface import GNNRunnerConfig, parse_args, validate_interface_compatibility
from .processor import BatchProcessor, load_structure_list

logger = logging.getLogger(__name__)

# Validate interface compatibility at module load time
validate_interface_compatibility()


def main() -> None:
    """Main entry point for GNN runner subprocess."""
    timestamp = datetime.now().strftime("%m-%dT%H:%M:%S")

    # Parse and convert to config
    args = parse_args()
    config = GNNRunnerConfig.from_args(args)

    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logging(
        log_file=log_dir / f"xx-gnn-runner-{timestamp}.log",
        console_level=logging.INFO,
        file_level=logging.INFO,
    )

    try:
        logger.info("=" * 60)
        logger.info("GNN Runner Starting")
        logger.info(f"Energy source: {config.energy_source}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Structure list: {config.structure_list}")
        logger.info(f"Progress interval: {config.progress_interval}")

        # Load structure list
        logger.info("Loading structure list...")
        structure_paths = load_structure_list(config.structure_list)

        if not structure_paths:
            logger.error("No valid structure paths found")
            sys.exit(1)

        # Create energy source
        try:
            energy_source = EnergySource(config.energy_source)
        except ValueError as e:
            logger.error(f"Invalid energy source '{config.energy_source}': {e}")
            sys.exit(1)

        # Create calculator once for entire batch
        logger.info("Initializing calculator...")
        calculator = create_calculator(energy_source, **config.to_calc_kwargs())

        # Process batch
        logger.info("Starting batch processing...")
        processor = BatchProcessor(calculator, config)
        processor.process_batch(structure_paths)

        logger.info("GNN runner completed successfully")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("GNN runner interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"GNN runner failed: {type(e).__name__}: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
