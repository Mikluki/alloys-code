import logging
from pathlib import Path

from vsf.core.hull.gnn_relax import (
    build_input_manifest,
    gnn_cache_to_eos_point,
    run_single_gnn_point,
)
from vsf.core.hull.parser import build_eos_table, eos_to_dataframe
from vsf.logging import setup_logging

LOGGER = setup_logging(
    console_level=logging.DEBUG,
    file_level=logging.DEBUG,
)

# Define paths
vasp_root = Path("00-test")  # Directory containing VASP calculations

LOGGER.info("=" * 80)
LOGGER.info("GNN Relax Pipeline - Smoke Test")
LOGGER.info("=" * 80)

# Step 1: Build input manifest from VASP directory
LOGGER.info("\n[1] Building input manifest from VASP...")
try:
    input_manifest = build_input_manifest(vasp_root)
    LOGGER.info(f"  ✓ Found {len(input_manifest)} structures")
    for point in input_manifest[:3]:  # Show first 3
        LOGGER.info(f"    - {point.vasp_dir_name} → {point.poscar_path}")
except Exception as e:
    LOGGER.error(f"  ✗ Error: {e}")
    exit(1)

# Step 2: Build VASP EOS table for comparison
LOGGER.info("\n[2] Building VASP EOS table...")
try:
    vasp_eos = build_eos_table(vasp_root, method="vasp")
    LOGGER.info(f"  ✓ Parsed {len(vasp_eos)} VASP points")
    LOGGER.info(
        f"    Status breakdown: {sum(1 for p in vasp_eos if p.status == 'ok')} ok, "
        f"{sum(1 for p in vasp_eos if p.status == 'failed')} failed, "
        f"{sum(1 for p in vasp_eos if p.status == 'missing')} missing"
    )
except Exception as e:
    LOGGER.error(f"  ✗ Error: {e}")
    exit(1)

# Step 3: Run GNN points (requires calculator)
LOGGER.info("\n[3] Running GNN relaxations...")
if len(input_manifest) > 0:
    try:
        from vsf.calculators.custom import Mace_mpa_0

        calculator = Mace_mpa_0()
        calculator.initialize()

        gnn_eos_points = []

        for input_point in input_manifest:
            # for input_point in input_manifest[:1]:  # Test with first structure only for smoke test
            LOGGER.info(f"  Testing: {input_point.vasp_dir_name}")

            try:
                cache = run_single_gnn_point(
                    input_point,
                    calculator=calculator,
                    method=calculator.energy_source.value,
                    fmax=0.02,
                    force_recalc=False,
                )

                LOGGER.info(f"    Model info: {cache.model_info}")
                LOGGER.info(f"    Cache status: {cache.status}")

                if cache.status == "ok":
                    eos_point = gnn_cache_to_eos_point(cache)
                    gnn_eos_points.append(eos_point)
                    LOGGER.info(f"    ✓ Converted to EosPoint: {eos_point.method}")
                else:
                    LOGGER.warning(f"    ✗ Relaxation failed: {cache.reason}")

            except Exception as e:
                LOGGER.error(f"    ✗ Error during relaxation: {e}", exc_info=True)

        LOGGER.info(f"\n  ✓ Successfully processed {len(gnn_eos_points)} GNN points")

    except Exception as e:
        LOGGER.error(f"  ✗ Error initializing calculator: {e}", exc_info=True)
else:
    LOGGER.info("  (No input points found)")

# Step 4: Convert VASP to DataFrame for analysis
LOGGER.info("\n[4] Converting VASP EOS to DataFrame...")
try:
    df = eos_to_dataframe(vasp_eos)
    LOGGER.info(f"  ✓ DataFrame shape: {df.shape}")
    LOGGER.info("\n  Sample rows:")
    LOGGER.info(
        df[["method", "vasp_dir_name", "volume_factor", "E", "status"]]
        .head(3)
        .to_string()
    )
except Exception as e:
    LOGGER.error(f"  ✗ Error: {e}")

LOGGER.info("\n" + "=" * 80)
LOGGER.info("Smoke test complete!")
LOGGER.info("=" * 80)
