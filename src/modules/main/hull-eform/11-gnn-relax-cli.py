"""
GNN EOS Relaxation Pipeline

Orchestrates:
1. VASP EOS table parsing
2. Input manifest construction from VASP calculations
3. GNN relaxation for all volume points
4. Conversion to unified EosPoint format for analysis
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from vsf.calculators.custom import (
    Allegro,
    Dpa31,
    Esen,
    Grace2lOamL,
    Mace_mpa_0,
    Nequip,
    Orbv3,
    SevenNet,
)
from vsf.core.hull.gnn_relax import (
    build_input_manifest,
    gnn_cache_to_eos_point,
    run_single_gnn_point,
)
from vsf.core.hull.parser import build_eos_table, eos_to_dataframe
from vsf.logging import setup_logging

# ============================================================================
# Configuration (modify for direct execution)
# ============================================================================

# Directories
VASP_ROOT = Path("00-test")
OUTPUT_DIR = Path("20-results/")

# GNN settings - change here for direct execution
CALCULATOR_CLASS = Mace_mpa_0
FMAX = 0.02

# Control
FORCE_RECALC = False
LIMIT_STRUCTURES = None  # Set to integer to limit (e.g., 3 for testing)

# ============================================================================
# Calculator Registry
# ============================================================================

CALCULATOR_REGISTRY = {
    "mace": Mace_mpa_0,
    "orbv3": Orbv3,
    "sevennet": SevenNet,
    "esen": Esen,
    "nequip": Nequip,
    "allegro": Allegro,
    "dpa31": Dpa31,
    "grace2loaml": Grace2lOamL,
}

# ============================================================================
# Setup
# ============================================================================

LOGGER = setup_logging(
    console_level=logging.DEBUG,
    file_level=logging.DEBUG,
)


def get_calculator_venv(calculator_class):
    """Get venv path from calculator instance. Returns None if not found."""
    try:
        calculator = calculator_class()
        calculator.initialize()
        venv_path = calculator.energy_source.venv
        return venv_path
    except Exception as e:
        LOGGER.error(f"Failed to get venv for {calculator_class.__name__}: {e}")
        return None


def main(calculator_class=None):
    """Main entry point for GNN EOS relaxation pipeline.

    Args:
        calculator_class: Calculator class to use. If None, uses CALCULATOR_CLASS from config.
    """
    if calculator_class is None:
        calculator_class = CALCULATOR_CLASS

    method_name = calculator_class.__name__
    calc_name = method_name.lower().replace("_mpa_0", "").replace("2loaml", "")

    # Validate paths
    if not VASP_ROOT.exists():
        LOGGER.error(f"VASP root directory does not exist: {VASP_ROOT}")
        return 1

    LOGGER.info("=" * 80)
    LOGGER.info("GNN EOS Relaxation Pipeline")
    LOGGER.info("=" * 80)
    LOGGER.info(f"VASP root: {VASP_ROOT}")
    LOGGER.info(f"Method: {method_name}")
    LOGGER.info(f"Calculator: {calculator_class.__name__}")
    LOGGER.info(f"Force threshold: {FMAX} eV/Å")
    if LIMIT_STRUCTURES:
        LOGGER.info(f"Limiting to first {LIMIT_STRUCTURES} structures")

    # =========================================================================
    # Step 1: Parse VASP EOS table
    # =========================================================================
    LOGGER.info("\n[1/4] Parsing VASP EOS table...")
    try:
        vasp_eos = build_eos_table(VASP_ROOT, method="vasp")
        LOGGER.info(f"  ✓ Parsed {len(vasp_eos)} VASP calculations")
        ok_count = sum(1 for p in vasp_eos if p.status == "ok")
        failed_count = sum(1 for p in vasp_eos if p.status == "failed")
        missing_count = sum(1 for p in vasp_eos if p.status == "missing")
        LOGGER.info(
            f"    Status: {ok_count} ok, {failed_count} failed, {missing_count} missing"
        )
    except Exception as e:
        LOGGER.error(f"  ✗ Failed to parse VASP EOS: {e}", exc_info=True)
        return 1

    # =========================================================================
    # Step 2: Build input manifest from VASP POSCARs
    # =========================================================================
    LOGGER.info("\n[2/4] Building input manifest...")
    try:
        input_manifest = build_input_manifest(VASP_ROOT)
        LOGGER.info(f"  ✓ Found {len(input_manifest)} structures with POSCAR")

        if LIMIT_STRUCTURES:
            input_manifest = input_manifest[:LIMIT_STRUCTURES]
            LOGGER.info(f"  → Limited to first {LIMIT_STRUCTURES} structures")
    except Exception as e:
        LOGGER.error(f"  ✗ Failed to build manifest: {e}", exc_info=True)
        return 1

    # =========================================================================
    # Step 3: Initialize calculator and run GNN relaxations
    # =========================================================================
    LOGGER.info("\n[3/4] Running GNN relaxations...")
    try:
        calculator = calculator_class()
        calculator.initialize()

        LOGGER.info(f"  ✓ Initialized {calculator_class.__name__}")
        LOGGER.info(f"    Model info: {calculator.get_model_info()}")

    except Exception as e:
        LOGGER.error(f"  ✗ Failed to initialize calculator: {e}", exc_info=True)
        return 1

    # Run relaxations
    gnn_eos_points = []
    failed_points = []

    for idx, input_point in enumerate(input_manifest, 1):
        LOGGER.info(f"\n  [{idx}/{len(input_manifest)}] {input_point.vasp_dir_name}")

        try:
            cache = run_single_gnn_point(
                input_point,
                calculator=calculator,
                method=method_name,
                fmax=FMAX,
                force_recalc=FORCE_RECALC,
            )

            if cache.status == "ok":
                # Convert to EosPoint and collect
                eos_point = gnn_cache_to_eos_point(cache)
                gnn_eos_points.append(eos_point)
                LOGGER.info(
                    f"      E: {cache.final_energy_eV:.6f} eV, F_max: {cache.max_force_eV_A:.3e} eV/Å, steps: {cache.n_steps}"
                )
            else:
                failed_points.append((input_point.vasp_dir_name, cache.reason))
                LOGGER.warning(f"      Failed: {cache.reason}")

        except Exception as e:
            failed_points.append((input_point.vasp_dir_name, str(e)))
            LOGGER.error(f"      Error: {e}", exc_info=True)

    LOGGER.info(
        f"\n  ✓ Processed {len(gnn_eos_points)} successful, {len(failed_points)} failed"
    )

    # =========================================================================
    # Step 4: Analysis and output
    # =========================================================================
    LOGGER.info("\n[4/4] Preparing analysis outputs...")

    try:
        # Convert both to DataFrames
        vasp_df = eos_to_dataframe(vasp_eos)
        gnn_df = eos_to_dataframe(gnn_eos_points)

        LOGGER.info(f"  ✓ VASP EOS: {len(vasp_df)} points")
        LOGGER.info(f"  ✓ GNN EOS: {len(gnn_df)} points")

        # Show summary
        LOGGER.info("\n  VASP summary:")
        LOGGER.info(
            f"    Volume range: {vasp_df['V'].min():.3f} - {vasp_df['V'].max():.3f} Å³"
        )
        LOGGER.info(
            f"    Energy range: {vasp_df['E'].min():.6f} - {vasp_df['E'].max():.6f} eV"
        )

        if len(gnn_df) > 0:
            LOGGER.info("\n  GNN summary:")
            LOGGER.info(
                f"    Energy range: {gnn_df['E'].min():.6f} - {gnn_df['E'].max():.6f} eV"
            )

        # Save outputs with calculator subdirectory
        if OUTPUT_DIR:
            results_dir = OUTPUT_DIR / calc_name
            results_dir.mkdir(parents=True, exist_ok=True)

            # Save CSVs
            vasp_csv = results_dir / "vasp_eos.csv"
            vasp_df.to_csv(vasp_csv, index=False)
            LOGGER.info(f"\n  ✓ Saved VASP EOS: {vasp_csv}")

            if len(gnn_df) > 0:
                gnn_csv = results_dir / f"gnn_{method_name}_eos.csv"
                gnn_df.to_csv(gnn_csv, index=False)
                LOGGER.info(f"  ✓ Saved GNN EOS: {gnn_csv}")

                # Compute and save energy differences
                try:
                    # Simple comparison: merge by vasp_dir_name and volume_factor
                    merged = pd.merge(
                        vasp_df[["vasp_dir_name", "volume_factor", "E"]].rename(
                            columns={"E": "E_vasp"}
                        ),
                        gnn_df[["vasp_dir_name", "volume_factor", "E"]].rename(
                            columns={"E": "E_gnn"}
                        ),
                        on=["vasp_dir_name", "volume_factor"],
                        how="inner",
                    )
                    merged["delta_E"] = merged["E_gnn"] - merged["E_vasp"]

                    comparison_csv = results_dir / f"comparison_{method_name}.csv"
                    merged.to_csv(comparison_csv, index=False)
                    LOGGER.info(f"  ✓ Saved comparison: {comparison_csv}")
                    LOGGER.info(f"    Mean ΔE: {merged['delta_E'].mean():.6f} eV")
                    LOGGER.info(
                        f"    RMS ΔE: {(merged['delta_E'] ** 2).mean() ** 0.5:.6f} eV"
                    )
                except Exception as e:
                    LOGGER.warning(f"  Could not compute comparison: {e}")

        # Report failures
        if failed_points:
            LOGGER.warning(f"\n  Failed relaxations ({len(failed_points)}):")
            for name, reason in failed_points[:5]:  # Show first 5
                LOGGER.warning(f"    - {name}: {reason[:60]}")
            if len(failed_points) > 5:
                LOGGER.warning(f"    ... and {len(failed_points) - 5} more")

    except Exception as e:
        LOGGER.error(f"  ✗ Failed to prepare outputs: {e}", exc_info=True)
        return 1

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Pipeline complete!")
    LOGGER.info("=" * 80)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GNN EOS relaxation pipeline for specified calculator"
    )
    parser.add_argument(
        "--calculator",
        type=str,
        choices=sorted(CALCULATOR_REGISTRY.keys()),
        help="Calculator to use (if not provided, uses CALCULATOR_CLASS from config)",
    )
    parser.add_argument(
        "--get-venv",
        type=str,
        choices=sorted(CALCULATOR_REGISTRY.keys()),
        help="Print venv path for calculator and exit (for bash orchestrator)",
    )

    args = parser.parse_args()

    # Handle --get-venv mode
    if args.get_venv:
        calc_class = CALCULATOR_REGISTRY[args.get_venv]
        venv = get_calculator_venv(calc_class)
        if venv:
            print(venv)
            sys.exit(0)
        else:
            sys.exit(1)

    # Run main
    if args.calculator:
        calc_class = CALCULATOR_REGISTRY[args.calculator]
        exit_code = main(calculator_class=calc_class)
    else:
        exit_code = main()

    sys.exit(exit_code)
