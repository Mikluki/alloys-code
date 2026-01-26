"""
VACF (Velocity Auto-Correlation Function) computation.
Computes normalized VACF on stitched trajectory (all-atoms pooled + species-resolved).
Outputs: 4 × .txt files + summary JSON + diagnostic plot.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

# ============================================================================
# CONFIG
# ============================================================================

TRAJ_PATH = Path("results/trajectories/AlCuNi_L1915_1400_chain_s01-s05_steps2865.traj")
OUTPUT_DIR = Path("results/vacf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECALCULATE = False  # Set to True to force recomputation (skip cache)
CACHE_FILE = OUTPUT_DIR / "vacf_cache.json"

POTIM = 2.0  # fs, VASP timestep
MAX_LAG = 2865  # all frames

# Slope fitting window
SLOPE_WINDOW_LAGS = 10  # fit over first 10 lags

# Integral window (short-time correlation proxy)
INTEGRAL_TAU_PS = 0.5  # ps

# Plot settings
DPI = 300

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================================
# FUNCTIONS
# ============================================================================


def load_trajectory(path):
    """
    Load trajectory from ASE .traj file.
    Returns: velocities [n_frames × n_atoms × 3] in Å/fs
    """
    log.info(f"Loading trajectory: {path}")
    traj = read(path, index=":")
    if not isinstance(traj, list):
        traj = [traj]

    n_frames = len(traj)
    n_atoms = len(traj[0])
    velocities = np.zeros((n_frames, n_atoms, 3))

    for i, frame in enumerate(traj):
        if frame.get_velocities() is None:
            raise ValueError(f"Frame {i} has no velocities")
        velocities[i] = frame.get_velocities()

    log.info(f"Loaded {n_frames} frames × {n_atoms} atoms")
    return velocities, traj[0]  # return velocities + reference frame (for species info)


def compute_vacf_raw(velocities, max_lag):
    """
    Compute raw velocity auto-correlation: <v(0)·v(t)>.

    Args:
        velocities: [n_frames × n_atoms × 3]
        max_lag: maximum lag to compute

    Returns:
        raw_corr: [max_lag] array of unnormalized correlations
    """
    log.info(f"Computing VACF raw correlations (max_lag={max_lag})")
    n_frames, n_atoms = velocities.shape[:2]
    max_lag = min(max_lag, n_frames)

    raw_corr = np.zeros(max_lag)

    for lag in range(max_lag):
        # <v(0)·v(lag)> averaged over all atoms and all valid time origins
        n_valid = n_frames - lag
        dot_products = np.sum(
            velocities[:n_valid] * velocities[lag:], axis=2
        )  # [n_valid × n_atoms]
        raw_corr[lag] = np.mean(dot_products)

    log.info(f"VACF raw computed: {len(raw_corr)} lags")
    return raw_corr


def normalize_vacf(raw_corr):
    """
    Normalize VACF: C(t)/C(0).

    Returns:
        vacf: [n_lags] normalized curve (should start at 1.0)
    """
    c0 = raw_corr[0]
    if c0 == 0:
        raise ValueError("C(0) is zero; velocities may be degenerate")
    vacf = raw_corr / c0
    return vacf


def fit_initial_slope(vacf, window_lags, potim):
    """
    Fit initial slope using OLS on first window_lags points.

    Args:
        vacf: normalized VACF curve
        window_lags: number of lags to fit over
        potim: timestep in fs

    Returns:
        slope_inv_ps: slope in 1/ps
        fit_window_str: human-readable string of fit window
    """
    x = np.arange(window_lags)  # lag indices
    y = vacf[:window_lags]

    # OLS: fit y = m*x + b
    coeffs = np.polyfit(x, y, 1)
    slope_per_lag = coeffs[0]

    # Convert to 1/ps: slope_per_lag is per lag; 1 lag = potim fs = potim/1000 ps
    slope_inv_ps = slope_per_lag / (potim / 1000.0)

    # Fit window in time
    t_end_fs = (window_lags - 1) * potim
    fit_window_str = f"0–{t_end_fs:.1f} fs ({window_lags} lags)"

    log.info(f"Initial slope: {slope_inv_ps:.4f} 1/ps (fit window: {fit_window_str})")
    return slope_inv_ps, fit_window_str


def find_first_zero_crossing(vacf, potim):
    """
    Find first zero-crossing of normalized VACF using linear interpolation.

    Returns:
        t0_ps: first zero-crossing time in ps (or None if no crossing found)
    """
    sign_changes = np.where(np.diff(np.sign(vacf)) != 0)[0]

    if len(sign_changes) == 0:
        log.warning("No zero-crossing found in VACF")
        return None

    idx_before = sign_changes[0]
    idx_after = idx_before + 1

    c_before = vacf[idx_before]
    c_after = vacf[idx_after]

    # Linear interpolation: t* = t_before + (0 - C_before) / (C_after - C_before) * dt
    frac = -c_before / (c_after - c_before)
    lag_crossing = idx_before + frac
    t0_ps = lag_crossing * potim / 1000.0

    log.info(f"First zero-crossing at {t0_ps:.4f} ps (lag index: {lag_crossing:.2f})")
    return t0_ps


def integrate_vacf(vacf, tau_ps, potim):
    """
    Compute short-time integral: ∫_0^τ C(t)/C(0) dt using trapezoidal rule.

    Args:
        vacf: normalized VACF curve
        tau_ps: integration limit in ps
        potim: timestep in fs

    Returns:
        integral_ps: integral value (in ps, dimensionless × time)
    """
    tau_fs = tau_ps * 1000.0
    lag_tau = int(np.ceil(tau_fs / potim))
    lag_tau = min(lag_tau, len(vacf))

    vacf_window = vacf[:lag_tau]
    integral_ps = np.trapz(vacf_window, dx=potim / 1000.0)  # convert dx to ps

    log.info(f"Short-time integral (0–{tau_ps} ps): {integral_ps:.4f} ps")
    return integral_ps


def save_vacf_cache(vacf_curves, metadata):
    """
    Save VACF curves (normalized) to cache file.

    Args:
        vacf_curves: dict {species_name: vacf_array}
        metadata: dict with trajectory info, etc.
    """
    cache_data = {
        "vacf_all_atoms": vacf_curves["all_atoms"].tolist(),
        "vacf_Al": vacf_curves["Al"].tolist(),
        "vacf_Cu": vacf_curves["Cu"].tolist(),
        "vacf_Ni": vacf_curves["Ni"].tolist(),
        "metadata": {
            "timestamp": str(np.datetime64("now")),
            "num_frames": metadata["num_frames"],
            "potim_fs": POTIM,
            "max_lag": metadata["max_lag"],
            "trajectory": str(TRAJ_PATH),
        },
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=2)

    log.info(f"VACF cache saved to {CACHE_FILE}")


def load_vacf_cache():
    """
    Load VACF curves from cache file.

    Returns:
        vacf_curves: dict {species_name: vacf_array}
        metadata: dict with trajectory info
        or (None, None) if cache doesn't exist or RECALCULATE is True
    """
    if RECALCULATE or not CACHE_FILE.exists():
        if RECALCULATE:
            log.info("RECALCULATE=True, skipping cache")
        else:
            log.info(f"Cache not found at {CACHE_FILE}")
        return None, None

    try:
        with open(CACHE_FILE, "r") as f:
            cache_data = json.load(f)

        vacf_curves = {
            "all_atoms": np.array(cache_data["vacf_all_atoms"]),
            "Al": np.array(cache_data["vacf_Al"]),
            "Cu": np.array(cache_data["vacf_Cu"]),
            "Ni": np.array(cache_data["vacf_Ni"]),
        }

        metadata = cache_data["metadata"]
        log.info(f"VACF cache loaded from {CACHE_FILE}")
        return vacf_curves, metadata

    except Exception as e:
        log.warning(f"Failed to load cache: {e}; will recompute")
        return None, None


def compute_all_scalars(vacf, potim, slope_window_lags, integral_tau_ps):
    """
    Compute all three scalars: initial slope, first zero-crossing, short-time integral.

    Returns:
        dict with keys: initial_slope_inv_ps, slope_fit_window, first_zero_crossing_ps, integral_ps
    """
    slope, slope_window_str = fit_initial_slope(vacf, slope_window_lags, potim)
    t0 = find_first_zero_crossing(vacf, potim)
    integral = integrate_vacf(vacf, integral_tau_ps, potim)

    return {
        "initial_slope_inv_ps": slope,
        "slope_fit_window": slope_window_str,
        "first_zero_crossing_ps": t0 if t0 is not None else "N/A",
        "integral_0_to_tau_ps": integral,
    }


def save_vacf_txt(vacf, potim, output_path):
    """
    Save VACF curve to .txt file: time (fs) | VACF (normalized).
    """
    n_lags = len(vacf)
    times_fs = np.arange(n_lags) * potim
    data = np.column_stack((times_fs, vacf))
    np.savetxt(
        output_path, data, fmt="%.6f", header="time_fs VACF_normalized", comments=""
    )
    log.info(f"Saved VACF to {output_path}")


def plot_vacf_summary(results, output_path):
    """
    Plot all VACF curves (all-atoms + 3 species) with zero-crossing markers.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"all_atoms": "black", "Al": "red", "Cu": "blue", "Ni": "green"}

    for species_name, color in colors.items():
        if species_name in results["vacf_curves"]:
            vacf = results["vacf_curves"][species_name]
            times_fs = np.arange(len(vacf)) * POTIM
            ax.plot(times_fs, vacf, label=species_name, color=color, linewidth=1.5)

            # Mark first zero-crossing
            scalars = results["scalars"][species_name]
            if scalars["first_zero_crossing_ps"] != "N/A":
                t0_fs = scalars["first_zero_crossing_ps"] * 1000.0
                ax.axvline(t0_fs, color=color, linestyle="--", alpha=0.5, linewidth=0.8)

    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("VACF (normalized)")
    ax.set_title("Velocity Auto-Correlation Function")
    ax.set_xlim(-10, 1000)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    log.info(f"Plot saved to {output_path} (DPI={DPI})")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================


def main():
    log.info("=" * 70)
    log.info("VACF Computation")
    log.info("=" * 70)

    # Try to load from cache first
    vacf_curves_cached, cache_metadata = load_vacf_cache()

    if vacf_curves_cached is not None:
        # Cache hit: use cached VACF curves
        log.info("Using cached VACF curves")
        vacf_curves = vacf_curves_cached
        n_frames = cache_metadata["num_frames"]
    else:
        # Cache miss: load trajectory and compute VACF
        log.info("Computing VACF from trajectory")
        velocities, ref_frame = load_trajectory(TRAJ_PATH)
        n_frames, n_atoms = velocities.shape[:2]

        # Get atomic numbers for species masks
        atomic_nums = ref_frame.get_atomic_numbers()
        mask_all = np.ones(n_atoms, dtype=bool)
        mask_al = atomic_nums == 13
        mask_cu = atomic_nums == 29
        mask_ni = atomic_nums == 28

        log.info(
            f"Atomic composition: Al={np.sum(mask_al)}, Cu={np.sum(mask_cu)}, Ni={np.sum(mask_ni)}"
        )

        species_specs = [
            ("all_atoms", mask_all),
            ("Al", mask_al),
            ("Cu", mask_cu),
            ("Ni", mask_ni),
        ]

        # Compute VACF for each species
        vacf_curves = {}
        for species_name, mask in species_specs:
            log.info(f"\n--- Computing VACF for {species_name} ---")

            # Mask velocities
            velocities_masked = velocities[:, mask, :]

            # Compute VACF
            raw_corr = compute_vacf_raw(velocities_masked, MAX_LAG)
            vacf = normalize_vacf(raw_corr)

            vacf_curves[species_name] = vacf

        # Save cache
        cache_metadata = {"num_frames": n_frames, "max_lag": MAX_LAG}
        save_vacf_cache(vacf_curves, cache_metadata)

    # Compute scalars (always, regardless of cache)
    log.info("\n--- Computing scalars ---")
    results = {"scalars": {}, "vacf_curves": vacf_curves}

    species_names = ["all_atoms", "Al", "Cu", "Ni"]
    for species_name in species_names:
        scalars = compute_all_scalars(
            vacf_curves[species_name], POTIM, SLOPE_WINDOW_LAGS, INTEGRAL_TAU_PS
        )
        results["scalars"][species_name] = scalars

        # Save VACF .txt
        output_txt = OUTPUT_DIR / f"vacf_{species_name}.txt"
        save_vacf_txt(vacf_curves[species_name], POTIM, output_txt)

    # Save summary JSON
    summary_dict = {
        "all_atoms": results["scalars"]["all_atoms"],
        "Al": results["scalars"]["Al"],
        "Cu": results["scalars"]["Cu"],
        "Ni": results["scalars"]["Ni"],
        "metadata": {
            "num_frames": n_frames,
            "potim_fs": POTIM,
            "slope_window_lags": SLOPE_WINDOW_LAGS,
            "integral_tau_ps": INTEGRAL_TAU_PS,
            "trajectory": str(TRAJ_PATH),
            "output_dir": str(OUTPUT_DIR),
        },
    }

    summary_json_path = OUTPUT_DIR / "vacf_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    log.info(f"Summary saved to {summary_json_path}")

    # Plot
    plot_path = OUTPUT_DIR / "vacf_plot.png"
    plot_vacf_summary(results, plot_path)

    log.info("\n" + "=" * 70)
    log.info("VACF computation complete")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
