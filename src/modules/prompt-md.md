# Physical Formulation

## Physical Formulation: GNN MD Validation Experiment

**Objective:** Validate that a GNN force field (e.g., MACE) reproduces liquid-phase ensemble properties of a reference DFT calculator (VASP) under identical conditions.

---

### Reference System: VASP NVT MD

**Setup:**

- Ensemble: NVT (Nosé-Hoover thermostat, fixed volume)
- Temperature: T = 1400 K
- System: Liquid Al-Cu-Ni alloy (~510 atoms)
- Timestep: Δt = 2 fs
- Duration: ~350 ps (multiple continued runs on cluster)

**Output:**

- Atomic trajectory: positions r_i(t) for all atoms, all timesteps
- Energies: E_pot(t) for each step
- Implicit: forces F_i(t) via VASP DFT

---

### GNN MD from Seed Frames

**Selection:**

- Extract K seed frames from VASP's equilibrated window (t > 5 ps burn-in, T stable ≈ 1400 K)
- Space seeds by Δt_seed ≈ 5 ps to decorrelate initial conditions
- Goal: sample the phase space where VASP found the system at equilibrium

**Simulation:**

- For each seed: run GNN MD under identical ensemble conditions
- Duration: up to 100 ps or until collapse
- Thermostat: NoseHooverChainNVT to match VASP's Nosé-Hoover dynamics
- Temperature control: same T_target = 1400 K, same damping τ_damp ≈ 200 fs

---

### Stability Monitoring

**Track collapse via indicators:**

| Indicator | Threshold  | Meaning                                        |
| --------- | ---------- | ---------------------------------------------- |
| F_max     | > 100 eV/Å | Force magnitude becomes physically unrealistic |
| r_min     | < 1.5 Å    | Atoms overlap (unphysical compression)         |
| ΔE spike  | > 5 eV     | Sudden energy jump (force field breakdown)     |

**Outcome:** Each GNN run produces either:

1. **Stable trajectory** — runs to full 100 ps without collapse
2. **Collapse event** — stable until time t_fail, then diverges

---

### Ensemble Comparison (Healthy Windows Only)

For time windows where both VASP and GNN trajectories are physically stable:

**Structural metrics:**

- **Radial Distribution Function** g*α*β(r) for each pair type
  - First peak position: characteristic bond distance
  - Coordination number ∫g(r)dr: local structure
  - Agreement → same local packing

**Dynamical metrics:**

- **Mean Squared Displacement** MSD(t) = ⟨|r_i(t) - r_i(0)|²⟩
  - Diffusion coefficient D from slope: MSD = 6Dt
  - Agreement → same atomic mobility

**Thermodynamic metrics:**

- **Temperature statistics:** ⟨T⟩, σ_T
  - Should match target (1400 K ± fluctuations)
- **Potential energy stats:** ⟨E_pot⟩, σ_E
  - Different calculators → different absolute energies, but σ_E should be similar
  - Autocorrelation time: how fast energy decorrelates

---

### Success Criteria

**Minimum** (Phase 1):

1. GNN runs for > 5–10 ps without immediate collapse
2. When stable, local structure (RDF peaks) within 10% of VASP
3. Stability distribution across K seeds: most complete the run

**Strong** (Phase 2):

1. GNN matches VASP ensemble averages to within 5%
2. Diffusion coefficients agree
3. All K seeds show consistent behavior
4. Collapse happens only in outlier seeds (if at all)

# Code

## Fix :: add stride to save traj

- a configurable **trajectory stride** (e.g. store every Nth frame), or

## Fix :: MSD implementation

Right now:

```python
ref_positions = window_frames[0].positions
for i, atoms in enumerate(window_frames):
    displacements = atoms.positions - ref_positions
    msd_array[i] = np.mean(np.sum(displacements**2, axis=1))
```

That’s a **single-origin** MSD (t₀ = first frame). True MSD averages over many time origins:

$$
\text{MSD}(t) = \left\langle |\mathbf{r}(t_0 + t) - \mathbf{r}(t_0)|^2 \right\rangle_{t_0}
$$

What you have is fine as a cheap first approximation (especially in a diffusive regime), but:

- for short windows / noisy data you might get quite a bit of variance,
- if you care about good diffusion coefficients, multi-origin averaging is better.

Still, the structure (compute MSD, fit linear region to get D) is correct.

If you want a minimal upgrade:

- pick a handful of origins (e.g. 5 evenly spaced in window), average MSD(t) over them.


## Fix :: load many XDATCAR, not 1

VASP writes trajectory continuations when:

- Simulation is very long (> 100k steps)
- Multiple job submissions (restarts)
- File size limits or I/O optimization

This creates fragmented data:

```
XDATCAR.1  (frames 0–1000)
XDATCAR.2  (frames 1000–2000)
XDATCAR.3  (frames 2000–3000)
...
```

**Goal**: Load all fragments, merge into continuous trajectory, then sample seeds.

---

### Solution Architecture

#### 1. **New Method in VASPTrajectoryAnalyzer**

```python
def load_trajectories_from_files(self, vasp_dir: Path) -> dict:
    """
    Load and merge fragmented XDATCAR.1, XDATCAR.2, ... files.

    Handles:
    - Detection of all XDATCAR.N files
    - Loading in correct order
    - Removing frame overlaps (VASP repeats last frame)
    - Merging temperature/energy time series
    - Maintaining time continuity

    Returns: {
        "frames": list[Atoms],  # Concatenated trajectory
        "temperatures": list[float],
        "energies": list[float],
        "times_ps": list[float],
    }
    """
```

#### 2. **Key Implementation Details**

- **Detect files**: `list(vasp_dir.glob("XDATCAR.*"))` sorted numerically
- **Load each**: `ase_read(xdatcar_i_path, index=":")`
- **Handle overlaps**: VASP often repeats last frame; detect & remove via small timestamp check
- **Merge times**: Accumulate time offsets from each file
- **Extract metadata**: Try OUTCAR for T/E data; fallback to defaults

#### 3. **Updated main.py**

```python
## OLD: Single file
vasp_data = vasp_analyzer.load_trajectory(vasp_dir)

## NEW: Multiple files (auto-detects)
vasp_data = vasp_analyzer.load_trajectories_from_files(vasp_dir)

## Rest proceeds identically
eq_start, eq_end = vasp_analyzer.identify_equilibrated_window(vasp_data, eq_config)
seeds = vasp_analyzer.select_seed_frames(...)
```

---

### Edge Cases to Handle

1. **Frame overlaps**: If XDATCAR.i ends at frame N and XDATCAR.i+1 starts at frame N (same atomic positions), deduplicate
2. **File ordering**: Must sort numerically (XDATCAR.10 comes after XDATCAR.9, not after XDATCAR.1)
3. **Missing OUTCAR**: Fallback to defaults (already handled in current code)

## Seed definition

A **seed** is a single atomic configuration (frame) from VASP's equilibrated trajectory used as the starting point for an independent GNN MD simulation.

**Key insight**:

- Each seed = **different snapshot in time** from VASP's equilibrium
- Running multiple seeds = sampling **different regions of phase space** at the same temperature
- Averaging across seeds = proper **ensemble statistics**

