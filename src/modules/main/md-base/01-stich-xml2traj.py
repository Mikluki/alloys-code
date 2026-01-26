"""
Stitch audit: evaluate boundary continuity between MD trajectory fragments.
Minimal version: read edges, check sanity, evaluate boundaries, output JSON.
"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
from ase.io import read

warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Configuration
TARGET_DIR = Path("data/AlCuNi_L1915_1400")
OUTPUT_DIR = Path("results/stitch_audit")

THRESHOLDS = {
    "cell_diff_tol": 0.05,
    "vol_diff_tol": 0.01,
    "rms_stitch": 0.05,
    "rms_no_stitch": 0.1,
}


def read_segment_edges(vasprun_path):
    """Read first and last frame from vasprun.xml."""
    atoms_first = read(vasprun_path, index=0)
    atoms_last = read(vasprun_path, index=-1)
    images = read(vasprun_path, index=":")
    nsteps = len(images)

    return {
        "path": str(vasprun_path),
        "nsteps": nsteps,
        "natoms": len(atoms_first),
        "symbols": atoms_first.get_chemical_symbols(),
        "Z": atoms_first.get_atomic_numbers(),
        "cell_first": atoms_first.cell[:],
        "cell_last": atoms_last.cell[:],
        "pos_first": atoms_first.positions.copy(),
        "pos_last": atoms_last.positions.copy(),
        "vel_first": atoms_first.get_velocities(),
        "vel_last": atoms_last.get_velocities(),
    }


def compute_pbc_displacement(pos_last, cell_last, pos_first, symbols, Z):
    """Compute minimal-image PBC displacement."""
    frac_last = np.linalg.solve(cell_last.T, pos_last.T).T
    frac_first = np.linalg.solve(cell_last.T, pos_first.T).T

    df = frac_first - frac_last
    df_wrapped = df - np.round(df)
    dr_cart = df_wrapped @ cell_last
    dr_norm = np.linalg.norm(dr_cart, axis=1)

    rms = np.sqrt(np.mean(dr_norm**2))
    max_disp = np.max(dr_norm)
    p95 = np.percentile(dr_norm, 95)

    # Top-20 largest displacements
    top_indices = np.argsort(dr_norm)[::-1][:20]
    topk_info = [
        {
            "atom_idx": int(idx),
            "symbol": symbols[idx],
            "dr_norm": float(dr_norm[idx]),
        }
        for idx in top_indices
    ]

    return {
        "rms": float(rms),
        "max": float(max_disp),
        "p95": float(p95),
        "topk": topk_info,
    }


def eval_boundary(seg_i, seg_i_plus_1, thresholds):
    """Evaluate boundary between two segments."""
    # Cell metrics
    cell_diff = np.linalg.norm(seg_i["cell_last"] - seg_i_plus_1["cell_first"])
    vol_i = np.linalg.det(seg_i["cell_last"])
    vol_ip1 = np.linalg.det(seg_i_plus_1["cell_first"])
    vol_diff_frac = abs(vol_i - vol_ip1) / abs(vol_i)

    # PBC displacement
    disp_dict = compute_pbc_displacement(
        seg_i["pos_last"],
        seg_i["cell_last"],
        seg_i_plus_1["pos_first"],
        seg_i_plus_1["symbols"],
        seg_i_plus_1["Z"],
    )

    rms_disp = disp_dict["rms"]
    max_disp = disp_dict["max"]

    # Decision logic
    if (
        cell_diff > thresholds["cell_diff_tol"]
        or vol_diff_frac > thresholds["vol_diff_tol"]
    ):
        decision = "NO_STITCH"
    elif rms_disp < 0.001 and max_disp < 0.01:
        decision = "DUPLICATE_FRAME"
    elif rms_disp < thresholds["rms_stitch"]:
        decision = "STITCH"
    elif rms_disp > thresholds["rms_no_stitch"]:
        decision = "NO_STITCH"
    else:
        decision = "UNCERTAIN"

    return {
        "cell_diff": float(cell_diff),
        "vol_diff_frac": float(vol_diff_frac),
        "rms_disp": float(rms_disp),
        "max_disp": float(max_disp),
        "p95_disp": float(disp_dict["p95"]),
        "decision": decision,
        "topk_atoms": disp_dict["topk"],
    }


def check_velocity_sanity(segments):
    """Spot-check velocities exist and temperature proxy is sane."""
    for i, seg in enumerate(segments):
        # Spot-check indices: first, mid, last
        spot_indices = [0, seg["nsteps"] // 2, -1]

        for idx in spot_indices:
            atoms = read(seg["path"], index=idx)
            vel = atoms.get_velocities()
            if vel is None:
                LOGGER.warning(f"Segment {i}: no velocities at frame {idx}")

        # KE proxy: mean |v|^2
        ke_last = (
            np.mean(np.sum(seg["vel_last"] ** 2, axis=1))
            if seg["vel_last"] is not None
            else None
        )
        ke_first = (
            np.mean(np.sum(seg["vel_first"] ** 2, axis=1))
            if seg["vel_first"] is not None
            else None
        )

        if ke_last is not None and ke_first is not None:
            LOGGER.info(f"Segment {i}: KE_first={ke_first:.4f}, KE_last={ke_last:.4f}")

        # Log boundary KE ratio
        if i < len(segments) - 1 and ke_last is not None:
            ke_next = (
                np.mean(np.sum(segments[i + 1]["vel_first"] ** 2, axis=1))
                if segments[i + 1]["vel_first"] is not None
                else None
            )
            if ke_next is not None and ke_last > 0:
                ratio = ke_next / ke_last
                if ratio > 100 or ratio < 0.01:
                    LOGGER.warning(
                        f"Large KE jump at boundary {i}-{i + 1}: ratio={ratio:.2e}"
                    )
                else:
                    LOGGER.info(f"KE ratio {i}-{i + 1}: {ratio:.2f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Target: {TARGET_DIR}")
    LOGGER.info(f"Output: {OUTPUT_DIR}")

    # Discover vasprun files
    vasprun_files = sorted(
        TARGET_DIR.glob("vasprun*.xml"),
        key=lambda p: int(p.name.replace("vasprun", "").replace(".xml", "").lstrip("_"))
        if p.name != "vasprun.xml"
        else -1,
    )
    LOGGER.info(f"Found {len(vasprun_files)} segments")

    # Read all segment edges
    segments = []
    for i, vasprun_path in enumerate(vasprun_files):
        try:
            seg = read_segment_edges(vasprun_path)
            segments.append(seg)
            LOGGER.info(
                f"[{i}] {vasprun_path.name}: {seg['nsteps']} steps, {seg['natoms']} atoms"
            )
        except Exception as e:
            LOGGER.error(f"Failed to read {vasprun_path}: {e}")
            raise

    # Velocity sanity checks
    LOGGER.info("Checking velocity sanity...")
    check_velocity_sanity(segments)

    # Evaluate boundaries
    LOGGER.info("Evaluating boundaries...")
    boundaries = []
    for i in range(len(segments) - 1):
        bd = eval_boundary(segments[i], segments[i + 1], THRESHOLDS)
        boundaries.append(
            {
                "seg_i": i,
                "seg_ip1": i + 1,
                **bd,
            }
        )
        LOGGER.info(
            f"[{i} → {i + 1}] {bd['decision']:20s} | "
            f"rms={bd['rms_disp']:.6f} Å | max={bd['max_disp']:.6f} Å"
        )

    # Output
    summary = {
        "target_dir": str(TARGET_DIR),
        "num_segments": len(segments),
        "thresholds": THRESHOLDS,
        "segments": [
            {
                "idx": i,
                "path": seg["path"],
                "nsteps": seg["nsteps"],
                "natoms": seg["natoms"],
            }
            for i, seg in enumerate(segments)
        ],
        "boundaries": boundaries,
    }

    summary_path = OUTPUT_DIR / "stitch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info(f"Wrote {summary_path}")
    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
