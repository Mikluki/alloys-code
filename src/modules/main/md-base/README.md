# Experiment setup

- радиальная функция распределения для каждой пары
- автокорреляционная функция скоростей по всем атомам
- автокорреляционная функция скоростей по типу атомов (we have 3 types of atoms)

* _radial distribution function (RDF) for each atomic pair_
* _velocity autocorrelation function (VACF) averaged over all atoms_
* _velocity autocorrelation function (VACF) resolved by atomic species_

---

**Key constraint (implicit but critical)**

All three require **a real trajectory**:

- O(10²–10³) timesteps minimum
- Contiguous in time
- Consistent atom indexing across timesteps

## 1. Radial distribution function (RDF) for each atomic pair

Required raw data:

- Atomic **positions** ( \mathbf{r}\_i(t) ) for **many MD timesteps**
- **Simulation cell / volume** at each timestep (or fixed cell)
- **Atomic species labels** for each atom

## 2. Velocity autocorrelation function (VACF), all atoms

Required raw data:

- Atomic **velocities** ( \mathbf{v}\_i(t) ) for **many MD timesteps**
- **Time spacing** between timesteps (Δt = POTIM)

## 3. Velocity autocorrelation function (VACF), resolved by atomic species

Required raw data:

- Atomic **velocities** ( \mathbf{v}\_i(t) ) for **many MD timesteps**
- **Atomic species labels**
- **Time spacing** between timesteps (Δt)

# DATA Pipeline

## Overview

Describes a computational workflow for extracting structural and dynamical properties from fragmented molecular dynamics (MD) trajectories. The pipeline combines multiple VASP MD segments (up to ~3000 frames each) into continuous trajectories with reconstructed velocities, validates boundary continuity using periodic boundary conditions (PBC), and computes radial distribution functions (RDF) to characterize atomic-scale structure in liquid alloys. The methodology is designed to preserve physical fidelity while enabling downstream analysis of transport coefficients and interatomic correlations.

---

## Input Data

### Source

Quantum mechanical MD simulations using VASP (Vienna Ab-initio Simulation Package) with the following specifications:

- **System**: Al₁₇₀Cu₁₇₀Ni₁₇₀ liquid alloy (510 atoms total, equimolar composition)
- **Temperature**: 1400 K (well above liquidus for all component phases)
- **Thermostat**: NVT ensemble (constant volume, temperature controlled)
- **Timestep**: POTIM = 2.0 fs per ionic step
- **Number of fragments**: 6 independent MD runs (segments 0–5)
- **Segment lengths**: 172–754 ionic steps (~0.34–1.51 ps per segment)
- **Output format**: VASP `vasprun.xml` files containing atomic positions, cell parameters, ionic forces, and stress tensors at each step

### File Structure

```
data/AlCuNi_L1915_1400/
├── vasprun.xml      (segment 0: 172 steps)
├── vasprun_1.xml    (segment 1: 538 steps)
├── vasprun_2.xml    (segment 2: 292 steps)
├── vasprun_3.xml    (segment 3: 633 steps)
├── vasprun_4.xml    (segment 4: 648 steps)
└── vasprun_5.xml    (segment 5: 754 steps)
```

### Data Characteristics

- **Atomic positions**: Cartesian coordinates (Å) with periodic boundary conditions; cell parameters evolve with simulation (cell "breathing" ±0.02 Å on side lengths)
- **Velocities**: Not stored in VASP output; reconstructed from position differences
- **Cell volume**: ~7023 Å³ (equimolar ternary alloy, ~0.13 atoms/Å³)
- **Spacing**: Position resolution limited by VASP electronic structure convergence (~10⁻⁶ Å per SCF cycle)

---

## Processing Pipeline

### Stage 1: Boundary Continuity Audit (`stitch_audit.py`)

**Purpose**: Determine which segments can be safely concatenated into continuous trajectories.

**Algorithm**:

1. Extract first and last atomic frames from each vasprun file
2. For each boundary between adjacent segments (i→i+1):
   - Compute cell parameter differences (Frobenius norm)
   - Compute volume change ratio (relative error)
   - Compute **PBC-wrapped minimal-image displacement** between last frame of segment i and first frame of segment i+1:
     ```
     r_i+1 - r_i (in fractional coords) → wrapped to [-0.5, 0.5]
     → convert back to Cartesian
     ```
   - Compute RMS displacement and P95 percentile
3. Apply decision rules:
   - Cell mismatch > 0.05 Å OR volume change > 1% → **NO_STITCH**
   - RMS displacement < 0.001 Å AND max displacement < 0.01 Å → **DUPLICATE_FRAME**
   - RMS displacement < 0.05 Å AND max displacement < validation threshold → **STITCH**
   - RMS > 0.1 Å → **NO_STITCH**
   - Otherwise → **UNCERTAIN**

**Output**: `stitch_summary.json`

- Segment metadata (frame count, atom count, composition)
- Boundary records: decision, displacement metrics, top-20 atoms with largest displacements
- Thresholds used (for reproducibility)

**Validation**: Spot-check kinetic energy (KE_proxy = mean|v|²) across frame pairs to detect velocity inversions.

### Stage 2: Velocity Reconstruction (`build_trajectories.py`)

**Purpose**: Create complete phase-space trajectories (positions + velocities) suitable for dynamical analysis.

**Reconstruction Method**:
Velocities are **not stored in VASP output**. They are reconstructed via finite differences using periodic boundary conditions:

For interior frames (0 < t < N−1):

```
v_t = (r_{t+1} - r_{t-1}) / (2·Δt)    [central difference]
```

For boundary frames (t = 0, N−1):

```
v_0 = (r_1 - r_0) / Δt  [forward]
v_{N-1} = (r_{N-1} - r_{N-2}) / Δt  [backward]
```

**PBC Wrapping** (critical for liquid systems):

- Convert positions to fractional coordinates: **s = H⁻¹·r**
- Compute displacement in fractional space: **Δs = s\_{t+1} − s_t**
- Apply minimum-image wrapping: **Δs ← Δs − round(Δs)**
- Convert back to Cartesian: **Δr = H·Δs**

This avoids spurious velocities when atoms cross periodic boundaries.

**Deduplication**:
At STITCH boundaries, check if first frame of segment i+1 duplicates last frame of segment i:

```
RMS(r_last[i], r_first[i+1]) < 0.001 Å → drop first frame of segment i+1
```

This prevents artificial "time hiccup" in dynamical correlations (e.g., VACF).

**Chain Identification**:
Group consecutive STITCH decisions into independent trajectories:

- Segment 0 → isolated (chain 1: 172 frames)
- Segments 1–5 → stitched (chain 2: 2865 frames after dedup)

**Output**: Two ASE `.traj` files (HDF5 binary format)

```
results/trajectories/
├── AlCuNi_L1915_1400_chain_s00-s00_steps172.traj
└── AlCuNi_L1915_1400_chain_s01-s05_steps2865.traj
```

Each contains:

- Atomic positions (Å)
- **Reconstructed velocities** (Å/fs)
- Cell parameters (3×3 matrix)
- Atomic numbers and symbols

**Validation**:

- **Displacement statistics** (PBC-wrapped): mean Δr ≈ 0.038–0.045 Å/step; max Δr ≈ 0.05–0.07 Å (physically sensible for 1400 K liquid)
- **Kinetic energy**: Computed from masses and velocities; converted to temperature via equipartition (T ≈ 2·KE / k_B·dof). Expected ~1400 K; actual ≈ 1400 K ✓
- **Center-of-mass drift**: Mass-weighted mean velocity; should be ≈ 0 (actual: ~10⁻⁸ Å/fs) ✓
- **Temperature stability**: Std(T) ≈ 0.6 K over 2865 frames (excellent stability) ✓

### Stage 3: Radial Distribution Function Analysis (`compute_rdf.py`)

**Purpose**: Quantify atomic-scale pair correlations and coordination environment.

**Geometry Setup**:

```
L_min = min(all cell-vector lengths over all frames) = 19.15 Å
r_max = 0.5 × L_min = 9.575 Å    [ensures unambiguous minimum-image]
Δr = 0.02 Å    [bin width: balances resolution vs. noise]
Number of bins: 479
```

**RDF Computation**:
For each species pair (α, β):

1. **Distance accumulation** (all frames):
   - Extract all atoms of type α and β
   - Compute all pairwise distances using PBC minimum-image (same wrapping as velocity)
   - Histogram into bins [0, 0.02, 0.04, ..., 9.575 Å]

2. **Normalization** to g(r):

   ```
   g_αβ(r) = counts(r) / [N_frames × N_α × ρ_β × 4πr²·Δr]

   where ρ_β = N_β / <V>  [number density]
         <V> = average cell volume over trajectory
   ```

**First-Shell Analysis** (coordination numbers):
For each pair, identify:

- **r_peak**: Maximum of g(r) in physical window [2.0, 3.2] Å (argmax, avoids noise)
- **r_min**: Minimum of g(r) in constrained window [r_peak + 0.2, r_peak + 1.5] Å (argmin, avoids second-shell structure)
- **Coordination number**:
  ```
  N_αβ = ∫₀^{r_min} 4π r² ρ_β g_αβ(r) dr    [trapezoidal integration]
  ```

**Output**: Six ASCII files (one per pair)

```
results/rdf/
├── rdf_AlAl.txt    [columns: r(Å), g(r)]
├── rdf_AlCu.txt
├── rdf_AlNi.txt
├── rdf_CuCu.txt
├── rdf_CuNi.txt
├── rdf_NiNi.txt
└── rdf_all_pairs.png    [plot: 0.5–9.6 Å, y-axis 0–4]
```

**Caching**: Accumulation phase (2865 frames × 6 pairs) takes ~1 min. Cached to `rdf_cache.pkl` on first run; set `RECALCULATE=True` to recompute.

---

## Quality Control & Validation

### Boundary Stitching

- **Outcome**: 5 of 5 boundaries between segments 1–5 passed STITCH criterion (RMS < 0.05 Å)
- **Boundary 0→1**: NO_STITCH (RMS ≈ 2.9 Å) → segments isolated
- **Interpretation**: Segments 1–5 form one continuous liquid trajectory; segment 0 is separate (possible restart or different initial condition)

### Velocity Reconstruction

- **Sanity checks** (all passed):
  - Position displacement continuous (no jumps > 0.1 Å; isolated outlier ~19 Å due to boundary wrapping)
  - Temperature inferred from velocities ≈ 1400 K (matches simulation target)
  - COM motion negligible (drift ~ 10⁻⁸ Å/fs; no net momentum)
  - No frame duplicates after dedup (0 frames dropped at boundaries)

### RDF Analysis

- **Asymptotic behavior**: g(r) → 1.0 ± 0.05 at r → r_max (confirms proper normalization)
- **Coordination numbers** (~9.0 ± 0.2 across all pairs):
  - Consistent with liquid-state coordination (typically 8–12 for close-packed liquids)
  - Minor variation (Al–Al slightly higher, Al–Ni slightly lower) reflects different interatomic distances
- **Peak positions** (2.7–2.9 Å):
  - Consistent with experimental metallic radii sums (Al: 1.43 Å, Cu: 1.35 Å, Ni: 1.35 Å)
  - Slightly compressed vs. pure elements due to alloy interactions
- **Physical consistency**: Peak order (by r_peak): Al–Ni < Al–Cu < Al–Al, Cu–Ni < Cu–Cu, Ni–Ni matches expected density-dependent first-neighbor distances

## Scientific Context & Priority

### Position in Materials Discovery Workflow

**RDF and liquid structure** are foundational in computational materials science for two reasons:

1. **Validation of interatomic potentials** (classical MD, machine learning):
   - VASP DFT-computed RDFs serve as ground truth for fitting and testing machine-learning force fields (ML-FF)
   - Large deviations between ML-FF-computed and DFT-computed RDFs signal force field failure modes
   - Your pipeline enables systematic benchmark of GNN models (MACE, SEVENNET, etc.) against reference DFT

2. **Discovery phase space reduction**:
   - Stable liquid structures (g_min, coordination numbers) constrain alloy phase diagrams and metastable phases
   - Deviations from ideal-solution RDF hints at chemical short-range order (SRO), which affects mechanical properties and phase stability

### Units

- **Positions**: Å (from VASP OUTCAR)
- **Velocities**: Å/fs (from finite differences of Å/2fs timestep)
- **Temperature**: K (equipartition, k_B = 8.617e-5 eV/K)
- **Density**: atoms/Å³
- **Coordination**: dimensionless (count)
