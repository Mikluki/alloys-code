"""
Decorrelate VASP MD trajectories.

Algorithm:
1) Detect burn-in index t0 from potential energy (pymbar.detect_equilibration).
2) Build a structural time series: fraction of atoms that moved ≥ δ Å between frames t and t−lag (PBC via minimum image).
3) Subsample post-burn-in frames using alchemlyb.statistical_inefficiency (fallback: fixed stride if estimator fails).
4) Provide sanity plots for the structural series.

Dependencies: ase, numpy, pandas, matplotlib, pymbar, alchemlyb.
Units: positions in Å; energies in eV; time index is frame number.
"""

import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alchemlyb.preprocessing.subsampling import statistical_inefficiency as alch_statinf
from ase.geometry import find_mic
from ase.io import read, write
from pymbar.timeseries import detect_equilibration
from pymbar.timeseries import statistical_inefficiency as g_est
from pymbar.utils import ParameterError


# ---------------------
# Core utilities
# ---------------------
def _mic_disp(R2: np.ndarray, R1: np.ndarray, cell, pbc) -> np.ndarray:
    """Return minimum-image displacement R2-R1 for given cell and PBC.

    Parameters
    ----------
    R2, R1 : (N,3) arrays
        Cartesian positions (Å).
    cell : (3,3) array-like
        Cell vectors (Å).
    pbc : (3,) bool
        Periodic boundary flags.

    Returns
    -------
    (N,3) array
        Minimum-image displacements (Å).
    """
    res = find_mic(R2 - R1, cell, pbc)
    return res[0] if isinstance(res, tuple) else res


def load_atoms() -> Tuple[list, str]:
    """Load trajectory as a list of ASE Atoms from vasprun.xml, OUTCAR, or XDATCAR.

    Returns
    -------
    atoms : list[ase.Atoms]
    source : str
        Filename used.

    Raises
    ------
    FileNotFoundError
        If none of the supported files are present.
    """
    for f in ("vasprun.xml", "OUTCAR", "XDATCAR"):
        if os.path.exists(f):
            return read(f, index=":"), f
    raise FileNotFoundError("No vasprun.xml / OUTCAR / XDATCAR found.")


def energy_series(atoms) -> Optional[pd.Series]:
    """Return potential energy per frame (eV) if available.

    Parameters
    ----------
    atoms : list[ase.Atoms]

    Returns
    -------
    pandas.Series or None
        Indexed by frame (name='time'). None if energies are not retrievable.
    """
    try:
        E = [a.get_potential_energy(apply_constraint=False) for a in atoms]
        return pd.Series(E, index=pd.Index(range(len(E)), name="time"), name="E_pot_eV")
    except Exception:
        return None


def fraction_moved_series(atoms, lag: int, delta: float) -> pd.Series:
    """Fraction of atoms with |Δr| ≥ delta between frames t and t−lag.

    Parameters
    ----------
    atoms : list[ase.Atoms]
    lag : int
        Frame separation (steps).
    delta : float
        Displacement threshold (Å).

    Returns
    -------
    pandas.Series
        Values in [0,1], indexed by frame t (name='time').
    """
    vals, idx = [], []
    for t in range(lag, len(atoms)):
        a2, a1 = atoms[t], atoms[t - lag]
        d = _mic_disp(
            a2.get_positions(), a1.get_positions(), a2.get_cell(), a2.get_pbc()
        )
        vals.append((np.linalg.norm(d, axis=1) >= delta).mean())
        idx.append(t)
    return pd.Series(
        vals, index=pd.Index(idx, name="time"), name=f"FracMoved_k{lag}_d{delta:.4f}"
    )


def subsample_structural(
    series: pd.Series, t0: int, conservative: bool, step_fallback: int
) -> Tuple[np.ndarray, float]:
    """Subsample frames post-burn-in using alchemlyb; fall back to fixed stride.

    Parameters
    ----------
    series : pandas.Series
        Structural time series (indexed by frame).
    t0 : int
        Burn-in start frame (inclusive).
    conservative : bool
        If True, integer spacing (fewer frames). If False, non-uniform spacing.
    step_fallback : int
        Stride used if statistical inefficiency cannot be estimated.

    Returns
    -------
    frames : ndarray[int]
        Selected frame indices (0-based).
    g : float
        Statistical inefficiency of the series after t0; NaN if not computed.
    """
    s_eq = series.loc[series.index >= t0]
    if len(s_eq) < 3 or float(np.nanstd(s_eq.values)) < 1e-12:
        return s_eq.index.values[:: max(1, step_fallback)], np.nan

    df_eq = pd.DataFrame({"frame": s_eq.index.values}, index=s_eq.index)
    try:
        df_sub = alch_statinf(
            df_eq,
            series=s_eq,
            drop_duplicates=True,
            sort=True,
            conservative=conservative,
        )
        frames = df_sub["frame"].to_numpy()
        try:
            g = float(g_est(s_eq.values))
        except Exception:
            g = np.nan
        return frames, g
    except ParameterError:
        return s_eq.index.values[:: max(1, step_fallback)], np.nan


def naive_acf(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """Compute a simple normalized ACF up to max_lag (no FFT, NaN-safe).

    Parameters
    ----------
    x : array-like
        Time series values.
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    ndarray
        ACF[0..max_lag]; ACF[0]=1.
    """
    x = np.asarray(x, float)
    x -= np.nanmean(x)
    var = np.nanvar(x)
    if var == 0:
        return np.r_[1.0, np.zeros(max_lag)]
    L = min(max_lag, len(x) - 2)
    acf = np.empty(L + 1)
    acf[0] = 1.0
    for k in range(1, L + 1):
        acf[k] = np.nanmean(x[:-k] * x[k:]) / var
    return acf


# ---------------------
# Public API
# ---------------------
def select_decorrelated_frames(
    atoms: list,
    *,
    lag: int = 10,
    delta: float = 0.005,
    conservative: bool = False,
    step_fallback: int = 10,
) -> Tuple[np.ndarray, Dict[str, Any], pd.Series]:
    """Return decorrelated frame indices using energy burn-in + fraction-moved thinning.

    Parameters
    ----------
    atoms : list[ase.Atoms]
    lag : int
        Frame separation for structural metric.
    delta : float
        Displacement threshold (Å).
    conservative : bool
        alchemlyb conservative mode.
    step_fallback : int
        Fallback stride if statistical inefficiency fails.

    Returns
    -------
    frames : ndarray[int]
        Selected frames (0-based).
    report : dict
        Diagnostics: t0, g_energy, g_struct, kept, total, lag, delta, flags.
    S : pandas.Series
        Fraction-moved structural series used for subsampling.
    """
    E = energy_series(atoms)
    if E is not None:
        t0, gE, NeffE = detect_equilibration(E.values)
    else:
        t0, gE, NeffE = 0, np.nan, np.nan

    S = fraction_moved_series(atoms, lag, delta)
    frames, gS = subsample_structural(S, t0, conservative, step_fallback)

    report = {
        "t0": int(t0),
        "g_energy": gE,
        "Neff_energy": NeffE,
        "g_struct": gS,
        "kept": int(len(frames)),
        "total": int(len(atoms)),
        "lag": lag,
        "delta": float(delta),
        "conservative": bool(conservative),
        "step_fallback": int(step_fallback),
        "source_has_energy": E is not None,
    }
    return frames, report, S


def plot_sanity(
    S: pd.Series, frames: np.ndarray, t0: int, title_prefix: str = ""
) -> None:
    """Plot structural series with kept frames, histogram, and ACF.

    Parameters
    ----------
    S : pandas.Series
        Structural series (fraction moved).
    frames : ndarray[int]
        Selected frames (0-based).
    t0 : int
        Burn-in start frame.
    title_prefix : str
        Optional title prefix (e.g., source file name).
    """
    # Time series
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(S.index, S.values, lw=1.5, label=S.name)    ax.axvline(t0, ls="--", c="red", label="Burn-in t₀")
    ax.scatter(frames, S.loc[frames], s=18, c="orange", zorder=3, label="Kept frames")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Fraction moved")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f"{title_prefix} Fraction-moved & selected frames")
    plt.tight_layout()
    plt.show()

    # Histogram
    S_eq = S.loc[S.index >= t0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(S_eq.values, bins=30, edgecolor="k")
    ax.set_xlabel("Fraction moved")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} Distribution (post burn-in)")
    plt.tight_layout()
    plt.show()

    # ACF
    acf = naive_acf(S_eq.values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(acf)), acf, lw=1.5)
    ax.axhline(1 / np.e, ls="--")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{title_prefix} ACF of fraction-moved (post burn-in)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------
# Example CLI usage
# ---------------------
if __name__ == "__main__":
    atoms, src = load_atoms()
    frames, rep, S = select_decorrelated_frames(
        atoms, lag=10, delta=0.005, conservative=False, step_fallback=10
    )
    print("REPORT:", rep)
    plot_sanity(S, frames, rep["t0"], title_prefix=os.path.basename(src))
    # write("decorrelated.xyz", [atoms[i] for i in frames])  # optional
