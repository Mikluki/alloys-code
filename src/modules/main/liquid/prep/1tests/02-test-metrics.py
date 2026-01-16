# pip install ase alchemlyb pymbar pandas numpy matplotlib
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alchemlyb.preprocessing.subsampling import statistical_inefficiency as alch_statinf
from ase.geometry import find_mic
from ase.io import read, write
from pymbar.timeseries import detect_equilibration
from pymbar.utils import ParameterError

# ----------------------------
# Tunables
# ----------------------------
LAG_SMALL = 3  # keep small to increase variability
DELTA_MOVE = 0.18  # lower so 'fraction moved' isn't all zeros
CONSERVATIVE = False
MIN_RMSD_GUARD = 0.0  # e.g. 0.35 Å to enforce extra geometric separation (0 disables)
STEP_FALLBACK = 10  # used only if series is (near) constant after t0
OUT_TRAJ = None  # e.g. "decorrelated.xyz"
INPUT_DIR = "Ni4_mp-23__supercell64_seed3"  # 79
INPUT_DIR = "Na_mp-10172__supercell64_seed3"


def load_traj():
    vasprun_path = f"{INPUT_DIR}/vasprun.xml"
    outcar_path = f"{INPUT_DIR}/OUTCAR"
    xdatcar_path = f"{INPUT_DIR}/XDATCAR"
    if os.path.exists(vasprun_path):
        return read(vasprun_path, index=":"), "vasprun.xml"
    if os.path.exists(outcar_path):
        return read(outcar_path, index=":"), "OUTCAR"
    if os.path.exists(xdatcar_path):
        return read(xdatcar_path, index=":"), "XDATCAR"
    raise FileNotFoundError("No vasprun.xml / OUTCAR / XDATCAR found.")


def energy_series(atoms) -> pd.Series:
    E = [a.get_potential_energy(apply_constraint=False) for a in atoms]
    return pd.Series(E, index=pd.Index(range(len(E)), name="time"), name="E_pot_eV")


def mic_disp(R2, R1, cell, pbc):
    res = find_mic(R2 - R1, cell, pbc)
    return res[0] if isinstance(res, tuple) else res


def series_rmsd_k(atoms, k: int) -> pd.Series:
    N = len(atoms)
    vals, idx = [], []
    for t in range(k, N):
        R2, R1 = atoms[t].get_positions(), atoms[t - k].get_positions()
        d = mic_disp(R2, R1, atoms[t].get_cell(), atoms[t].get_pbc())
        vals.append(np.sqrt(((d**2).sum(axis=1)).mean()))
        idx.append(t)
    return pd.Series(vals, index=pd.Index(idx, name="time"), name=f"RMSD_k{k}_A")


def series_mad_k(atoms, k: int) -> pd.Series:
    # mean absolute displacement per atom: mean(||d||)
    N = len(atoms)
    vals, idx = [], []
    for t in range(k, N):
        R2, R1 = atoms[t].get_positions(), atoms[t - k].get_positions()
        d = mic_disp(R2, R1, atoms[t].get_cell(), atoms[t].get_pbc())
        vals.append(np.linalg.norm(d, axis=1).mean())
        idx.append(t)
    return pd.Series(vals, index=pd.Index(idx, name="time"), name=f"MAD_k{k}_A")


def series_frac_moved_k(atoms, k: int, delta: float) -> pd.Series:
    N = len(atoms)
    vals, idx = [], []
    for t in range(k, N):
        R2, R1 = atoms[t].get_positions(), atoms[t - k].get_positions()
        d = mic_disp(R2, R1, atoms[t].get_cell(), atoms[t].get_pbc())
        moved = (np.linalg.norm(d, axis=1) >= delta).mean()
        vals.append(moved)  # 0..1 fraction
        idx.append(t)
    return pd.Series(
        vals, index=pd.Index(idx, name="time"), name=f"FracMoved_k{k}_d{delta:.2f}"
    )


def subsample_by_series(series: pd.Series, t0: int) -> Tuple[np.ndarray, float]:
    """
    Use alchemlyb thinning post-burn-in. If the series has ~zero variance or
    pymbar fails, fall back to simple slicing every STEP_FALLBACK frames.
    Returns (frames, g_estimate_or_nan).
    """
    s_eq = series.loc[series.index >= t0]
    if len(s_eq) < 3:
        return np.array([], dtype=int), np.nan

    # zero-variance or nearly constant? -> fallback slicing
    if (
        float(np.nanstd(s_eq.values)) < 1e-12
        or len(np.unique(np.round(s_eq.values, 12))) <= 2
    ):
        frames = s_eq.index.values[:: max(1, STEP_FALLBACK)]
        return frames, np.nan

    # normal path: alchemlyb/pymbar
    df_eq = pd.DataFrame({"frame": s_eq.index.values}, index=s_eq.index)
    try:
        df_sub = alch_statinf(
            df_eq,
            series=s_eq,
            drop_duplicates=True,
            sort=True,
            conservative=CONSERVATIVE,
        )
        frames = df_sub["frame"].to_numpy()
        # report g on same data (best-effort; if it fails, set nan)
        try:
            from pymbar.timeseries import statistical_inefficiency as g_est

            g = g_est(s_eq.values)
        except Exception:
            g = np.nan
        return frames, float(g)
    except ParameterError:
        # fallback: simple slicing
        frames = s_eq.index.values[:: max(1, STEP_FALLBACK)]
        return frames, np.nan


def rmsd_guard(atoms, frames: np.ndarray, delta: float) -> np.ndarray:
    if delta <= 0 or len(frames) <= 1:
        return frames
    kept = [frames[0]]
    last = frames[0]
    for f in frames[1:]:
        d = mic_disp(
            atoms[f].get_positions(),
            atoms[last].get_positions(),
            atoms[f].get_cell(),
            atoms[f].get_pbc(),
        )
        rmsd = np.sqrt(((d**2).sum(axis=1)).mean())
        if rmsd >= delta:
            kept.append(f)
            last = f
    return np.asarray(kept, dtype=int)


def main():
    atoms, src = load_traj()
    N = len(atoms)

    # Burn-in from energy if possible
    try:
        E = energy_series(atoms)
        t0, gE, NeffE = detect_equilibration(E.values)
        print(f"[Energy] t0={t0}, g≈{gE:.2f}, Neff≈{NeffE:.1f}")
    except Exception as e:
        print(f"[Energy] not available from {src}: {e}; using t0=0")
        t0, E = 0, None

    # Build structural series candidates (short-memory ones)
    S_cands: Dict[str, pd.Series] = {
        f"RMSD_k{LAG_SMALL}": series_rmsd_k(atoms, LAG_SMALL),
        f"MAD_k{LAG_SMALL}": series_mad_k(atoms, LAG_SMALL),
        f"FracMoved_k{LAG_SMALL}_d{DELTA_MOVE:.2f}": series_frac_moved_k(
            atoms, LAG_SMALL, DELTA_MOVE
        ),
    }

    results = {}
    for name, S in S_cands.items():
        frames, g = subsample_by_series(S, t0)
        frames2 = rmsd_guard(atoms, frames, MIN_RMSD_GUARD)
        results[name] = {"frames": frames2, "g": g, "series": S}

    # Pick the series that yields the most frames
    best_name = max(results.keys(), key=lambda k: len(results[k]["frames"]))
    best = results[best_name]
    kept = best["frames"]

    # Report all
    print("\n=== Structural thinning results (post burn-in) ===")
    for name, r in results.items():
        print(f"{name:>28s}: kept {len(r['frames']):4d} frames, g≈{r['g']:.2f}")
    print(
        f"\n>> Selected: {best_name} — kept {len(kept)} of {N} frames (source: {src})"
    )
    print("Indices (0-based):", kept.tolist())

    # Plot best series
    fig, ax = plt.subplots(figsize=(9, 4))
    S = best["series"]
    ax.plot(S.index, S.values, label=f"{best_name}")
    ax.scatter(kept, S.loc[kept], s=14, label="Kept frames")
    ax.axvline(t0, ls="--", c="red", label="Burn-in t0")
    ax.set_xlabel("Frame")
    ax.set_ylabel(best_name)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    if OUT_TRAJ:
        write(OUT_TRAJ, [atoms[i] for i in kept])
        print(f"Wrote {len(kept)} frames to {OUT_TRAJ}")


if __name__ == "__main__":
    main()
