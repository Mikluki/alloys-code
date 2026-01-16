"""
RADIAL DISTRIBUTION FUNCTION (RDF) CALCULATOR FOR PHASE IDENTIFICATION
=======================================================================

PURPOSE:
Identify fully liquid structures among POSCAR snapshots by analyzing RDF g(r).

PHYSICS:
- Crystalline: many narrow, tall peaks; deep valleys; oscillations persist.
- Liquid: broad first peak; damped oscillations; g(r) → 1.

WORKFLOW:
1) Compute RDF per POSCAR.
2) Extract a late "tail" slice (index-based, cell-agnostic).
3) Compute three features on the tail:
   - C: count of narrow, prominent peaks
   - F: fraction of deep valleys
   - M: mean |g-1|
4) Warn if ≥2 of {C≥3, F≥0.25, M≥0.18} are true (with a small C+M bonus).

OUTPUT:
No plots (prototype). Logs [WARN] for solid-like candidates.
"""

from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Lattice
from pymatgen.io.vasp import Poscar


# ---------- small utilities ----------
def _moving_average(y: np.ndarray, win: int = 5) -> np.ndarray:
    """Centered moving average; win is made odd. Use for peak finding only."""
    if win is None or win <= 1:
        return y
    win = int(win)
    if win % 2 == 0:
        win += 1
    k = np.ones(win, dtype=float) / win
    return np.convolve(y, k, mode="same")


def _local_maxima(y: np.ndarray) -> np.ndarray:
    if len(y) < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1


def _local_minima(y: np.ndarray) -> np.ndarray:
    if len(y) < 3:
        return np.array([], dtype=int)
    return np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1


def _first_peak_index(r: np.ndarray, g: np.ndarray, bin_width: float) -> int:
    """Find first peak: search 1.5–6 Å for tallest local maximum; fallback to global."""
    lo, hi = 1.5, min(6.0, float(r[-1]))
    i_lo = np.searchsorted(r, lo, side="left")
    i_hi = np.searchsorted(r, hi, side="right")
    if i_hi - i_lo < 5:
        return int(np.argmax(g))
    sub = g[i_lo:i_hi]
    locs = np.where((sub[1:-1] > sub[:-2]) & (sub[1:-1] > sub[2:]))[0] + 1
    if len(locs) == 0:
        return int(i_lo + int(np.argmax(sub)))
    best = locs[np.argmax(sub[locs])]
    return int(i_lo + int(best))


def _half_prominence_width(y: np.ndarray, i: int, half_level: float) -> int:
    """Return width in bins at the given level below the peak."""
    n = len(y)
    if n < 3:
        return 0
    # left
    L = i
    while L - 1 >= 0 and y[L] > half_level:
        L -= 1
    # right
    R = i
    while R + 1 < n and y[R] > half_level:
        R += 1
    return max(0, R - L)


# ---------- simplified, index-based tail slice ----------
def _tail_slice_by_index(r: np.ndarray, g: np.ndarray, *, bin_width: float) -> slice:
    """
    Robust tail window independent of cell size.
    - Find first peak index i1.
    - Start at max(i1 + ~1.0 Å, 50% of array).
    - Ensure length ≥ max(60 bins, 4 Å / dr).
    """
    N = len(r)
    if N < 10:
        return slice(N, N)  # empty
    i1 = _first_peak_index(r, g, bin_width)
    dr = float(np.median(np.diff(r))) if N > 1 else bin_width
    bins_1A = max(1, int(round(1.0 / max(dr, 1e-8))))
    start = max(i1 + bins_1A, int(0.5 * N))
    min_len = max(60, int(round(4.0 / max(dr, 1e-8))))  # at least 60 bins or 4 Å
    end = N
    if end - start < min_len:
        start = max(0, N - min_len)
    if start >= end:
        return slice(N, N)
    return slice(start, end)


# ---------- indicators on tail (C, F, M) ----------
def rdf_tail_features(
    r: np.ndarray,
    g: np.ndarray,
    *,
    bin_width: float = 0.1,
    smooth_win: int = 5,
    peak_height_gate: float = 1.10,  # g >= 1.1
    peak_prom_thresh: float = 0.30,  # prominence >= 0.3
    peak_width_bins_max: int = 4,  # <= 4 bins (narrow)
    valley_thresh: float = 0.70,  # minima <= 0.70
) -> tuple[dict, slice]:
    """
    Compute C (narrow-peak count), F (deep-valley fraction), M (mean |g-1|) on a robust tail.
    Returns (metrics, tail_slice).
    """
    # Smooth for peak/min detection only
    gs = _moving_average(g, smooth_win)
    tail = _tail_slice_by_index(r, gs, bin_width=bin_width)
    rt = r[tail]
    gt = g[tail]
    gst = gs[tail]

    if len(rt) < 8:
        return dict(C=0, F=0.0, M=0.0), tail

    # --- peaks ---
    C = 0
    p_idx = _local_maxima(gst)
    for ip in p_idx:
        y_peak = gst[ip]
        if y_peak < peak_height_gate:
            continue
        # simple local baseline = max(min left, min right) in a small flank
        flank = max(3, int(round(0.75 / max(np.median(np.diff(rt)), 1e-8))))  # ~0.75 Å
        L = max(0, ip - flank)
        R = min(len(gst), ip + flank + 1)
        left_min = np.min(gst[L:ip]) if ip > L else y_peak
        right_min = np.min(gst[ip + 1 : R]) if ip + 1 < R else y_peak
        base = max(left_min, right_min)
        prom = float(max(0.0, y_peak - base))
        if prom < peak_prom_thresh:
            continue
        half_level = y_peak - 0.5 * prom
        width_bins = _half_prominence_width(
            gst[L:R], i=ip - L, half_level=float(half_level)
        )
        if width_bins <= peak_width_bins_max:
            C += 1

    # --- valleys ---
    m_idx = _local_minima(gst)
    if len(m_idx) > 0:
        F = float(np.mean(gst[m_idx] <= valley_thresh))
    else:
        F = 0.0

    # --- roughness ---
    M = float(np.mean(np.abs(gt - 1.0)))

    return dict(C=int(C), F=F, M=M), tail


def classify_solid_like_simple(
    r: np.ndarray,
    g: np.ndarray,
    *,
    bin_width: float = 0.1,
    smooth_win: int = 5,
    # decision thresholds
    C_min: int = 3,
    F_min: float = 0.25,
    M_min: float = 0.18,
    # bonus coupling for borderline C
    bonus_C_if_M: float = 0.22,
    require_votes: int = 2,
) -> tuple[bool, dict, dict]:
    """
    Minimal rule-based classifier using three tail features.
    Returns (flag, votes, metrics) where flag=True means "solid-like".
    """
    metrics, tail = rdf_tail_features(r, g, bin_width=bin_width, smooth_win=smooth_win)
    votes = dict(
        peak_persistence=(metrics["C"] >= C_min)
        or (metrics["C"] >= 2 and metrics["M"] >= bonus_C_if_M),
        valley_depth=(metrics["F"] >= F_min),
        tail_roughness=(metrics["M"] >= M_min),
    )
    flag = sum(votes.values()) >= int(require_votes)
    # enrich metrics for logging
    tail_info = dict(
        tail_start=float(r[tail.start]) if len(r) else np.nan,
        tail_end=float(r[tail.stop - 1]) if len(r) else np.nan,
        tail_len=int(tail.stop - tail.start),
    )
    return flag, votes, {**metrics, **tail_info}


# ---------- RDF calculation (unchanged logic; plotting off by default) ----------


def calculate_rdf(
    poscar_path: Path,
    r_max: float = 20.0,
    bin_width: float = 0.1,
    save_plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast RDF for periodic structures with explicit image replication to r_max.
    Matches the counting convention of get_all_neighbors (pairs are counted
    per central atom), so normalization remains valid and results match your
    'old' working version — only faster.
    """
    # --- load structure ---
    p = Poscar.from_file(poscar_path)
    s = p.structure
    lattice = Lattice(s.lattice.matrix)
    frac = np.array([site.frac_coords for site in s], dtype=float)  # (N, 3)
    n_atoms = len(frac)

    # --- choose replication range so all images within r_max are included ---
    # Conservative cube based on edge lengths; small N so fine.
    a, b, c = lattice.abc
    nx = int(np.ceil(r_max / a))
    ny = int(np.ceil(r_max / b))
    nz = int(np.ceil(r_max / c))

    # fractional translation vectors
    shifts = np.array(
        list(product(range(-nx, nx + 1), range(-ny, ny + 1), range(-nz, nz + 1))),
        dtype=float,
    )  # (M, 3)
    # all translated images in fractional coords
    # For efficient vectorization, we'll compute distances per central atom,
    # to keep the same "double-count per central atom" convention.
    bins = np.arange(0.0, r_max + bin_width, bin_width)
    r_values = bins[:-1] + bin_width / 2.0
    hist = np.zeros_like(r_values, dtype=float)

    # Precompute Cartesian shift vectors for each fractional shift
    cart_shifts = shifts @ lattice.matrix  # (M, 3)

    # Cartesian positions of original cell
    cart0 = frac @ lattice.matrix  # (N, 3)

    # Loop over central atoms (N=64 → cheap) and accumulate neighbor distances
    for i in range(n_atoms):
        # positions of all images relative to atom i
        # Build all positions of other atoms in all images
        # cart positions of atoms in each image: cart0 + cart_shifts[m]
        # relative vectors: (cart0 + shift) - cart0[i]
        rel = (
            cart0[None, :, :] + cart_shifts[:, None, :] - cart0[i][None, None, :]
        )  # (M, N, 3)
        d = np.linalg.norm(rel, axis=-1).ravel()  # (M*N,)

        # exclude self at zero shift (distance ~0)
        # (there will be exactly one zero — atom i in the 0,0,0 image)
        d = d[(d > 1e-12) & (d <= r_max)]

        # histogram counts for this central atom
        h, _ = np.histogram(d, bins=bins)
        hist += h  # counting per central atom => double counts i<->j, matching expected

    # --- normalization identical to your old version ---
    volume = lattice.volume
    number_density = n_atoms / volume
    shell_vol = 4.0 * np.pi * r_values**2 * bin_width
    expected = number_density * shell_vol * n_atoms  # per central atom * n_atoms

    g_r = np.divide(hist, expected, out=np.zeros_like(hist, float), where=expected > 0)

    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(r_values, g_r, linewidth=1.5)
        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.title(f"RDF - {poscar_path.stem}")
        plt.grid(True, alpha=0.3)
        plt.xlim(0, r_max)
        plt.axhline(y=1.0, linestyle="--", alpha=0.5)
        out_dir = Path(poscar_path.parent.parent / "RDF-plots")
        out_dir.mkdir(exist_ok=True)
        plt.savefig(
            out_dir / f"rdf_{poscar_path.parent.stem}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    return r_values, g_r


# ---------- Batch ----------
def batch_rdf_analysis(
    poscar_paths: list[Path],
    save_plot: bool = False,
    **kwargs,
) -> dict:
    """
    Process multiple POSCAR files and log phase warnings.
    Returns dict {stem: (r, g_r)}.
    """
    results = {}
    for poscar_path in poscar_paths:
        try:
            r, g_r = calculate_rdf(poscar_path, save_plot=save_plot, **kwargs)

            # classify with simplified rules
            flag, votes, M = classify_solid_like_simple(
                r,
                g_r,
                bin_width=kwargs.get("bin_width", 0.1),
                smooth_win=5,
                # thresholds tuned to catch solids while sparing hot liquids
                C_min=3,
                F_min=0.25,
                M_min=0.18,
                bonus_C_if_M=0.22,
                require_votes=2,
            )

            if flag:
                print(
                    f"[WARN] {poscar_path} solid-like: "
                    f"C={M['C']}, F={M['F']:.2f}, M={M['M']:.2f}; "
                    f"votes={votes}; tail=[{M['tail_start']:.2f},{M['tail_end']:.2f}]Å n={M['tail_len']}"
                )
            else:
                print(
                    f"[OK]   {poscar_path}: "
                    f"C={M['C']}, F={M['F']:.2f}, M={M['M']:.2f}; votes={votes}"
                )

            results[poscar_path.stem] = (r, g_r)
        except Exception as e:
            print(f"Failed to process {poscar_path.stem}: {e}")
    return results


def analyze_poscar_path(p):
    # call your batch function with a single-element list
    return batch_rdf_analysis([p], save_plot=False, bin_width=0.1)


# ---------- CLI ----------

if __name__ == "__main__":
    import subprocess

    # base_dir = Path("x-test-minimal")
    base_dir = Path("x-test")
    element = "Au"

    result = subprocess.run(
        f"rg -u -l {element} {base_dir}/**/POSCAR",
        capture_output=True,
        text=True,
        shell=True,
    )
    poscar_paths = [Path(p) for p in result.stdout.strip().split("\n") if p]

    with ProcessPoolExecutor(max_workers=8) as ex:
        list(ex.map(analyze_poscar_path, poscar_paths))
