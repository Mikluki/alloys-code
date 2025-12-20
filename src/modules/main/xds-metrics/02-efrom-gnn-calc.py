import logging
from datetime import datetime
from pathlib import Path

from vsf.energy.energy_source import EnergySource
from vsf.energy.workflow_gnn import GNNEnergyWorkflow
from vsf.logging import setup_logging


def main():
    """Main batch processing script for multiple energy sources."""
    timestamp = datetime.now().strftime("%m-%dT%H:%M:%S")
    (log_dir := Path("logs")).mkdir(exist_ok=True)
    LOGGER = setup_logging(
        log_file=log_dir / f"x-eform_{timestamp}.log",
        console_level=logging.INFO,
        file_level=logging.INFO,
    )

    # Calculation args
    device = "cpu"  # Same device for all sources
    force_recalculate = False
    cleanup_deprecated = False
    formation_energy_only = True

    # Directories
    target_dirs = [Path("xout/x-liquid-k1-gnncp")]
    hull_dir = Path("xout/00-hull-gnncp")

    # Configuration
    energy_sources = [
        # EnergySource.MACE_MPA_0,
        # EnergySource.ORBV3,
        EnergySource.SEVENNET,
        EnergySource.ESEN_30M_OAM,
        # # cuda only
        # EnergySource.NEQUIP,
        # EnergySource.ALLEGRO,
        # # new
        # EnergySource.DPA31,
        # EnergySource.GRACE_2L_OAM_L,
    ]

    # Structure directories
    structure_dirs = [d for td in target_dirs for d in td.iterdir()]
    LOGGER.info(f"Found {len(structure_dirs)} structure directories")

    if not structure_dirs:
        LOGGER.error("No structure directories found. Exiting.")
        return

    # Initialize workflow once
    gnn_workflow = GNNEnergyWorkflow()

    # Process each energy source completely before moving to next
    all_summaries = {}

    for energy_source in energy_sources:
        try:
            # Step 1: Extract hull reference energies
            LOGGER.info(f"Extracting hull references for {energy_source.value}...")
            hull_ref = gnn_workflow.extract_hull_reference_energies(
                hull_dir=hull_dir,
                energy_source=energy_source,
                force_recalculate_hull=force_recalculate,
                cleanup_deprecated=cleanup_deprecated,
                device=device,
            )

            if not hull_ref:
                LOGGER.error(f"No hull references for {energy_source.value}")
                continue

            LOGGER.info(f"Hull references for {energy_source.value}: {hull_ref}")

            # Step 2: Process target structures
            LOGGER.info(f"Processing structures for {energy_source.value}...")
            summary = gnn_workflow.run_workflow(
                structure_dirs=structure_dirs,
                hull_reference=hull_ref,
                energy_source=energy_source,
                formation_energy_only=formation_energy_only,
                force_recalculate=force_recalculate,
                cleanup_deprecated=cleanup_deprecated,
                device=device,
            )

            all_summaries[energy_source.value] = summary

        except Exception as e:
            LOGGER.error(f"Failed {energy_source.value}: {e}")

    # Print overall summary
    LOGGER.info(f"\n🎉 Batch processing completed!")
    LOGGER.info(f"Logs written to: xeform-calc.log")


if __name__ == "__main__":
    main()
