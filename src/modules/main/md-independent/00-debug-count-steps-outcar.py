#!/usr/bin/env python3
"""
Count convergence markers and force blocks in VASP OUTCAR file.
Configure the target OUTCAR path at the top.
"""

# ============================================================================
# CONFIGURATION - Set your target OUTCAR file path here
# ============================================================================
OUTCAR_PATH = "/home/mik/1_projects/alloys_code/src/modules_v2/main/md/data/AlCuNi_L1915_1400/OUTCAR"
# ============================================================================

from pathlib import Path


def analyze_outcar(filepath):
    """Analyze OUTCAR file for convergence and force information."""
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"❌ Error: File not found: {filepath}")
        return None

    try:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read()

        # Count electronic convergence (SCF loop done)
        electronic_converged = content.count("aborting loop because EDIFF is reached")

        # Count ionic convergence (geometry/structural minimization done)
        ionic_converged = content.count(
            "reached required accuracy - stopping structural energy minimisation"
        )

        # Count force blocks
        force_blocks = content.count("TOTAL-FORCE (eV/Angst)")

        return {
            "electronic_converged": electronic_converged,
            "ionic_converged": ionic_converged,
            "force_blocks": force_blocks,
        }

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None


if __name__ == "__main__":
    print(f"Analyzing: {OUTCAR_PATH}")
    print("=" * 80)

    result = analyze_outcar(OUTCAR_PATH)

    if result:
        print("\n📊 Convergence Summary:")
        print(
            f"  ✓ Electronic convergence (EDIFF reached):  {result['electronic_converged']}"
        )
        print(
            f"  ✓ Ionic convergence (structural minimised): {result['ionic_converged']}"
        )
        print(f"\n📈 Force Calculations:")
        print(f"  ✓ TOTAL-FORCE blocks:                      {result['force_blocks']}")
        print("\n" + "=" * 80)
    else:
        print("Failed to analyze file.")
