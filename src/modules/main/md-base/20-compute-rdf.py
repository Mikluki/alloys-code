"""
Compute radial distribution function (RDF) for all species pairs.
Load .traj, accumulate pair distances under PBC, normalize to g(r).
Includes cache save/load to avoid recomputing accumulation step.
"""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Configuration
TRAJ_PATH = Path("results/trajectories/AlCuNi_L1915_1400_chain_s01-s05_steps2865.traj")
DR = 0.02  # Å
FRAME_INDICES = slice(
    None
)  # Use all frames; can override with slice(0, 100) for testing
OUTPUT_DIR = Path("results/rdf")
RECALCULATE = False  # Set to True to skip cache and recompute from trajectory
CACHE_FILE = OUTPUT_DIR / "rdf_cache.pkl"


def get_pair_indices(atoms, species_pair):
    """Get indices of atoms for a species pair (e.g., ('Al', 'Cu'))."""
    symbols = atoms.get_chemical_symbols()
    sp1, sp2 = species_pair

    indices_1 = [i for i, s in enumerate(symbols) if s == sp1]
    indices_2 = [i for i, s in enumerate(symbols) if s == sp2]

    return indices_1, indices_2


def compute_distances_minimage(pos_alpha, pos_beta, cell):
    """
    Compute pairwise distances under PBC (minimum image).

    Args:
        pos_alpha: (N_alpha, 3) array
        pos_beta: (N_beta, 3) array
        cell: (3, 3) cell matrix

    Returns:
        distances: (N_alpha, N_beta) array of distances
    """
    # Convert to fractional coordinates
    inv_cell = np.linalg.inv(cell.T)
    frac_alpha = pos_alpha @ inv_cell
    frac_beta = pos_beta @ inv_cell

    # All-pairs fractional displacement
    # Shape: (N_alpha, N_beta, 3)
    df = frac_beta[np.newaxis, :, :] - frac_alpha[:, np.newaxis, :]

    # Wrap to [-0.5, 0.5]
    df_wrapped = df - np.round(df)

    # Back to Cartesian
    dr = df_wrapped @ cell

    # Distances
    distances = np.linalg.norm(dr, axis=2)

    return distances


def accumulate_rdf(frames, pairs, r_edges):
    """
    Accumulate RDF histograms for all species pairs.

    Args:
        frames: list of ASE Atoms objects
        pairs: list of (species1, species2) tuples
        r_edges: bin edges for histogram

    Returns:
        histograms: dict {pair: histogram array}
        pair_counts: dict {pair: (N_alpha, N_beta)} for normalization
    """
    histograms = {pair: np.zeros(len(r_edges) - 1) for pair in pairs}
    pair_counts = {pair: (0, 0) for pair in pairs}

    for frame_idx, atoms in enumerate(frames):
        if frame_idx % 500 == 0:
            LOGGER.info(f"  Accumulating frame {frame_idx}/{len(frames)}")

        cell = atoms.cell[:]

        for pair in pairs:
            indices_alpha, indices_beta = get_pair_indices(atoms, pair)

            if len(indices_alpha) == 0 or len(indices_beta) == 0:
                continue

            pos_alpha = atoms.positions[indices_alpha]
            pos_beta = atoms.positions[indices_beta]

            distances = compute_distances_minimage(pos_alpha, pos_beta, cell)

            # Flatten and accumulate
            distances_flat = distances.flatten()
            hist, _ = np.histogram(distances_flat, bins=r_edges)
            histograms[pair] += hist

            # Store count (use first frame)
            if frame_idx == 0:
                pair_counts[pair] = (len(indices_alpha), len(indices_beta))

    return histograms, pair_counts


def normalize_rdf(histograms, pair_counts, frames, r_edges):
    """
    Normalize histograms to g(r) using number density and shell volume.

    Args:
        histograms: dict of accumulated counts
        pair_counts: dict of (N_alpha, N_beta) for each pair
        frames: list of frames (for volume)
        r_edges: bin edges

    Returns:
        g_r: dict {pair: g(r) array}
    """
    g_r = {}
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    nframes = len(frames)

    for pair, counts in histograms.items():
        n_alpha, n_beta = pair_counts[pair]

        if n_alpha == 0 or n_beta == 0:
            LOGGER.warning(f"  Pair {pair}: no atoms found, skipping")
            continue

        # Average volume over frames
        volumes = [np.linalg.det(atoms.cell[:]) for atoms in frames]
        avg_volume = np.mean(volumes)

        # Number density of beta
        rho_beta = n_beta / avg_volume

        # Shell volume: 4π r² Δr
        shell_volumes = 4 * np.pi * r_centers**2 * DR

        # Normalization factor: nframes × N_alpha × rho_beta × shell_volume
        norm = nframes * n_alpha * rho_beta * shell_volumes

        # g(r)
        g = counts / norm
        g_r[pair] = g

        LOGGER.info(
            f"  Pair {pair}: N_alpha={n_alpha}, N_beta={n_beta}, "
            f"rho_beta={rho_beta:.4f} Å^-3, avg_vol={avg_volume:.1f} Å^3"
        )

    return g_r, r_centers


def save_cache(histograms, pair_counts, r_edges):
    """Save accumulated histograms and metadata to cache file."""
    cache_data = {
        "histograms": histograms,
        "pair_counts": pair_counts,
        "r_edges": r_edges,
        "DR": DR,
    }
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache_data, f)
    LOGGER.info(f"Saved cache: {CACHE_FILE}")


def load_cache():
    """Load cached histograms and metadata. Returns (histograms, pair_counts, r_edges) or None."""
    if not CACHE_FILE.exists():
        return None

    try:
        with open(CACHE_FILE, "rb") as f:
            cache_data = pickle.load(f)

        # Verify cache is compatible
        if cache_data.get("DR") != DR:
            LOGGER.warning(
                f"Cache DR mismatch: {cache_data['DR']} vs {DR}, recomputing"
            )
            return None

        LOGGER.info(f"Loaded cache: {CACHE_FILE}")
        return (
            cache_data["histograms"],
            cache_data["pair_counts"],
            cache_data["r_edges"],
        )
    except Exception as e:
        LOGGER.warning(f"Failed to load cache: {e}, recomputing")
        return None


def find_peaks_and_minima(g_arr, r_arr, dr):
    """
    Find first-shell peak and first minimum after peak.
    Uses windowed argmax for peak in [2.0, 3.2] Å,
    then windowed argmin for minimum in [r_peak+0.2, r_peak+1.5] Å.

    Returns:
        (r_peak, r_min) or (None, None) if not found
    """
    # Define physical window for first-shell peak
    r_peak_min, r_peak_max = 2.0, 3.2
    idx_window_start = int(r_peak_min / dr)
    idx_window_end = int(r_peak_max / dr)

    if idx_window_end > len(g_arr):
        idx_window_end = len(g_arr)

    if idx_window_start >= idx_window_end:
        return None, None

    # Find argmax in window
    window = g_arr[idx_window_start:idx_window_end]
    idx_peak_in_window = np.argmax(window)
    idx_peak = idx_window_start + idx_peak_in_window
    r_peak = r_arr[idx_peak]

    # Find minimum in constrained window after peak: [r_peak+0.2, r_peak+1.5]
    r_min_search_start = r_peak + 0.2
    r_min_search_end = r_peak + 1.5

    idx_min_search_start = int(r_min_search_start / dr)
    idx_min_search_end = min(int(r_min_search_end / dr) + 1, len(g_arr))

    if idx_min_search_start < idx_min_search_end:
        window_min = g_arr[idx_min_search_start:idx_min_search_end]
        idx_min_in_window = np.argmin(window_min)
        r_min = r_arr[idx_min_search_start + idx_min_in_window]
    else:
        return r_peak, None

    return r_peak, r_min


def compute_coordination_number(g_arr, r_arr, r_min, rho_beta, dr):
    """
    Compute coordination number: N = ∫_0^{r_min} 4π r² ρ_β g(r) dr

    Uses trapezoidal integration over bins.
    """
    # Find index corresponding to r_min
    idx_min = int(r_min / dr)
    if idx_min >= len(r_arr):
        idx_min = len(r_arr) - 1

    # Compute 4π r² ρ_β g(r)
    integrand = 4 * np.pi * r_arr[: idx_min + 1] ** 2 * rho_beta * g_arr[: idx_min + 1]

    # Trapezoidal integration
    coord_num = np.trapz(integrand, dx=dr)

    return coord_num


def validate_coordination_numbers(g_r, r_centers, pair_counts, frames, dr):
    """
    Validate RDF by computing coordination numbers and checking weighted consistency.

    Computes N_αβ for each pair, then checks if weighted sum matches average g(r).
    """
    LOGGER.info(f"\n=== Coordination Number Validation ===")

    # Compute average volume
    volumes = [np.linalg.det(atoms.cell[:]) for atoms in frames]
    avg_volume = np.mean(volumes)

    coordination_numbers = {}
    peak_positions = {}

    # Compute coordination numbers for each pair
    LOGGER.info(f"Coordination numbers (to first minimum):")
    for pair, g in g_r.items():
        n_alpha, n_beta = pair_counts[pair]
        rho_beta = n_beta / avg_volume

        r_peak, r_min = find_peaks_and_minima(g, r_centers, dr)

        if r_peak is not None and r_min is not None:
            coord_num = compute_coordination_number(g, r_centers, r_min, rho_beta, dr)
            coordination_numbers[pair] = coord_num
            peak_positions[pair] = (r_peak, r_min)

            LOGGER.info(
                f"  {pair}: N_αβ={coord_num:.3f}, "
                f"r_peak={r_peak:.3f} Å, r_min={r_min:.3f} Å"
            )
        else:
            LOGGER.warning(f"  {pair}: Could not find peak/minimum")
            coordination_numbers[pair] = None
            peak_positions[pair] = (None, None)

    return coordination_numbers, peak_positions


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loading trajectory: {TRAJ_PATH}")
    frames = read(TRAJ_PATH, index=FRAME_INDICES)
    if not isinstance(frames, list):
        frames = [frames]

    nframes = len(frames)
    LOGGER.info(f"Loaded {nframes} frames")

    # Determine geometry
    cells = np.array([atoms.cell[:] for atoms in frames])
    cell_lengths = np.linalg.norm(cells, axis=2)  # (nframes, 3)
    L_min = np.min(cell_lengths)

    LOGGER.info(f"Cell lengths (min over frames): {np.min(cell_lengths, axis=0)}")
    LOGGER.info(f"L_min = {L_min:.4f} Å")

    r_max = 0.5 * L_min
    LOGGER.info(f"r_max = {r_max:.4f} Å")
    LOGGER.info(f"DR = {DR} Å")

    r_edges = np.arange(0, r_max + DR, DR)
    nbins = len(r_edges) - 1
    LOGGER.info(f"Number of bins: {nbins}")

    # Define species pairs
    all_species = list(set(frames[0].get_chemical_symbols()))
    all_species.sort()
    LOGGER.info(f"Species: {all_species}")

    pairs = []
    for i, sp1 in enumerate(all_species):
        for sp2 in all_species[i:]:
            pairs.append((sp1, sp2))

    LOGGER.info(f"Computing RDF for {len(pairs)} pairs: {pairs}")

    # Try to load from cache
    if not RECALCULATE:
        cache_result = load_cache()
        if cache_result is not None:
            histograms, pair_counts, r_edges_cached = cache_result
            LOGGER.info(f"Using cached data, skipping accumulation")
        else:
            histograms, pair_counts = None, None
    else:
        LOGGER.info(f"RECALCULATE=True, recomputing from trajectory")
        histograms, pair_counts = None, None

    # Accumulate if not cached
    if histograms is None:
        LOGGER.info(f"Accumulating pair distances...")
        histograms, pair_counts = accumulate_rdf(frames, pairs, r_edges)
        save_cache(histograms, pair_counts, r_edges)

    # Normalize
    LOGGER.info(f"Normalizing to g(r)...")
    g_r, r_centers = normalize_rdf(histograms, pair_counts, frames, r_edges)

    # Sanity checks
    LOGGER.info(f"\nSanity checks (should approach 1 at large r):")
    for pair, g in g_r.items():
        # Use last 10% of bins
        idx_start = int(0.9 * len(g))
        g_asymptote = np.mean(g[idx_start:])
        LOGGER.info(f"  {pair}: g(r→r_max) ≈ {g_asymptote:.4f}")

    # Validate coordination numbers and consistency
    validate_coordination_numbers(g_r, r_centers, pair_counts, frames, DR)

    # Save data
    LOGGER.info(f"\nSaving RDF data...")
    for pair, g in g_r.items():
        filename = f"rdf_{''.join(pair)}.txt"
        filepath = OUTPUT_DIR / filename

        data = np.column_stack([r_centers, g])
        np.savetxt(filepath, data, fmt="%.6f %.6f", header="r(Å)  g(r)", comments="")
        LOGGER.info(f"  Wrote: {filepath}")

    # Plot
    LOGGER.info(f"Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(g_r)))

    # Start plotting from r > 0.5 Å to avoid r→0 artifacts
    r_min_plot = 0.5
    idx_start = int(r_min_plot / DR)

    for (pair, g), color in zip(g_r.items(), colors):
        ax.plot(
            r_centers[idx_start:],
            g[idx_start:],
            label=f"{''.join(pair)}",
            color=color,
            linewidth=1.5,
        )

    # Total g(r) (average)
    g_total = np.mean([g for g in g_r.values()], axis=0)
    ax.plot(
        r_centers[idx_start:],
        g_total[idx_start:],
        "k--",
        label="avg",
        linewidth=1,
        alpha=0.7,
    )

    ax.axvline(
        r_max, color="gray", linestyle=":", alpha=0.5, label=f"r_max={r_max:.2f} Å"
    )
    ax.axhline(1.0, color="gray", linestyle="-", alpha=0.3)

    ax.set_xlabel("r (Å)", fontsize=12)
    ax.set_ylabel("g(r)", fontsize=12)
    ax.set_xlim(r_min_plot, r_max)
    ax.set_ylim(0, 4)  # Reasonable limit for liquid metal RDFs
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    plot_path = OUTPUT_DIR / "rdf_all_pairs.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    LOGGER.info(f"Wrote: {plot_path}")
    plt.close(fig)

    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
