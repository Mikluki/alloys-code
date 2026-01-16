import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alchemlyb.preprocessing.subsampling import statistical_inefficiency as alch_statinf
from ase.geometry import find_mic
from ase.io import read, write
from pymbar.timeseries import detect_equilibration
from pymbar.timeseries import statistical_inefficiency as g_est
from pymbar.utils import ParameterError

# ---- knobs ------------------------------------------------------
LAG_CANDIDATES = [3, 5, 8, 10]  # smaller lags -> more variability
DELTA_CANDIDATES = np.linspace(0.005, 0.08, 16)  # Å; scan from 0.005 upward
CONSERVATIVE = False  # non-uniform spacing → more kept frames
STEP_FALLBACK = 10  # used if pymbar cannot estimate g
WRITE_OUT = None  # e.g. "decorrelated.xyz"
DIR = "Na_mp-10172__supercell64_seed3"
# ----------------------------------------------------------------


def load_traj():
    for f in (f"{DIR}/vasprun.xml", f"{DIR}/OUTCAR", f"{DIR}/XDATCAR"):
        if os.path.exists(f):
            return read(f, index=":"), f
    raise FileNotFoundError("No vasprun.xml / OUTCAR / XDATCAR found.")


def mic_disp(R2, R1, cell, pbc):
    res = find_mic(R2 - R1, cell, pbc)
    return res[0] if isinstance(res, tuple) else res


def frac_moved_series(atoms, lag, delta):
    N = len(atoms)
    vals = []
    idx = []
    for t in range(lag, N):
        a2, a1 = atoms[t], atoms[t - lag]
        d = mic_disp(
            a2.get_positions(), a1.get_positions(), a2.get_cell(), a2.get_pbc()
        )
        vals.append((np.linalg.norm(d, axis=1) >= delta).mean())
        idx.append(t)
    return pd.Series(
        vals, index=pd.Index(idx, name="time"), name=f"FracMoved_k{lag}_d{delta:.3f}"
    )


def subsample(series, t0):
    s = series.loc[series.index >= t0]
    if len(s) < 3 or float(np.nanstd(s.values)) < 1e-12:
        return s.index.values[:: max(1, STEP_FALLBACK)], np.nan
    df = pd.DataFrame({"frame": s.index.values}, index=s.index)
    try:
        df_sub = alch_statinf(
            df, series=s, drop_duplicates=True, sort=True, conservative=CONSERVATIVE
        )
        frames = df_sub["frame"].to_numpy()
        try:
            g = float(g_est(s.values))
        except Exception:
            g = np.nan
        return frames, g
    except ParameterError:
        return s.index.values[:: max(1, STEP_FALLBACK)], np.nan


def naive_acf(x, max_lag=None):
    x = np.asarray(x, float)
    x -= np.nanmean(x)
    n = len(x)
    max_lag = min(n - 1, 200) if max_lag is None else max_lag
    var = np.nanvar(x)
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    if var == 0:
        return acf
    for k in range(1, max_lag + 1):
        acf[k] = np.nanmean(x[:-k] * x[k:]) / var
    return acf


def main():
    atoms, src = load_traj()
    N = len(atoms)
    print(f"Loaded {N} frames from {src}")

    # Burn-in from energy if available; else t0=0
    try:
        E = [a.get_potential_energy(apply_constraint=False) for a in atoms]
        t0, gE, NeffE = detect_equilibration(np.asarray(E))
        print(f"[Energy] t0={t0}, g≈{gE:.2f}, Neff≈{NeffE:.1f}")
    except Exception:
        t0, gE = 0, None
        print("[Energy] not available → t0=0")

    # Grid search (lag, delta)
    best = None
    for lag in LAG_CANDIDATES:
        for delta in DELTA_CANDIDATES:
            S = frac_moved_series(atoms, lag, delta)
            frames, gS = subsample(S, t0)
            kept = len(frames)
            # skip flat series (all zeros/ones)
            if np.nanstd(S.values) < 1e-12:
                continue
            rec = {
                "lag": lag,
                "delta": delta,
                "kept": kept,
                "g": gS,
                "frames": frames,
                "series": S,
            }
            print(
                f"[k={lag:>2}, δ={delta:>5.3f}] kept={kept:4d}, g≈{('nan' if np.isnan(gS) else round(gS,2))}"
            )
            if (
                (best is None)
                or (kept > best["kept"])
                or (kept == best["kept"] and lag < best["lag"])
            ):
                best = rec

    if best is None:
        print(
            "All scanned series were flat. Try smaller δ (≤0.005) and/or larger lag (e.g., 15–20)."
        )
        return

    lag, delta = best["lag"], best["delta"]
    S, frames, gS = best["series"], best["frames"], best["g"]
    print(
        f"\nSelected: k={lag}, δ={delta:.3f} Å → kept {len(frames)} / {N} (g≈{gS if not np.isnan(gS) else 'nan'})"
    )
    print("Frame indices:", frames.tolist())

    # ---- sanity plots ----
    # (1) Time series with kept frames and burn-in
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(S.index, S.values, lw=1.5, label=f"Frac moved ≥ {delta:.3f} Å (k={lag})")
    ax.axvline(t0, ls="--", c="red", label="Burn-in t₀")
    ax.scatter(frames, S.loc[frames], s=18, c="orange", zorder=3, label="Kept frames")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Fraction moved")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Fraction-moved & selected frames")
    plt.tight_layout()
    plt.show()

    # (2) Histogram post burn-in
    S_eq = S.loc[S.index >= t0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(S_eq.values, bins=30, edgecolor="k")
    ax.set_xlabel("Fraction moved")
    ax.set_ylabel("Count")
    ax.set_title("Distribution (post burn-in)")
    plt.tight_layout()
    plt.show()

    # (3) ACF post burn-in
    acf = naive_acf(S_eq.values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(acf)), acf, lw=1.5)
    ax.axhline(1 / np.e, ls="--")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("ACF of fraction-moved (post burn-in)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    if WRITE_OUT:
        write(WRITE_OUT, [atoms[i] for i in frames])
        print(f"Wrote {len(frames)} → {WRITE_OUT}")


if __name__ == "__main__":
    main()
