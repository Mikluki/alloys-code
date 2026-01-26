## MD trajectory stability benchmark

**Status summary (high level)**

* The dataset does **not** contain a statistically meaningful MD trajectory.
* We have **6 converged ionic configurations** of a 510-atom Al–Cu–Ni system.
* Each configuration includes reliable **DFT forces, cell stress, and potential energy**.
* The data is sufficient for **force/stress benchmarking** (e.g., GNN vs VASP) on a small set of realistic snapshots.
* The data is **not sufficient** for trajectory-based analyses (RDF, diffusion, time correlations, thermodynamic averages).

**Conclusion:**
This dataset should be treated as a **small benchmark set of independent snapshots**, not as molecular dynamics data.


## Hull materials project (Volume 0.8 - 1.2) Relaxation benchmark


```bash
Cu3Pd_mp-672265_0.8   :: CELL_DIFF ::          None , ATOM_DIFF ::  5.710886e-03
Cu3Pd_mp-672265_1     :: CELL_DIFF ::          None , ATOM_DIFF ::  5.135191e-03
Cu3Pd_mp-672265_1.2   :: CELL_DIFF ::          None , ATOM_DIFF ::  2.470133e-02
--
Ti3Al_mp-1823_0.8     :: CELL_DIFF ::          None , ATOM_DIFF ::  8.125116e-03
Ti3Al_mp-1823_1       :: CELL_DIFF ::          None , ATOM_DIFF ::          None
Ti3Al_mp-1823_1.2     :: CELL_DIFF ::          None , ATOM_DIFF ::  4.501617e-04
---
TiAl2_mp-567705_0.8   :: CELL_DIFF ::          None , ATOM_DIFF ::  4.822241e-02
TiAl2_mp-567705_1     :: CELL_DIFF ::          None , ATOM_DIFF ::  3.629036e-03
TiAl2_mp-567705_1.2   :: CELL_DIFF ::          None , ATOM_DIFF ::  8.278237e-01
---
TiAl3_mp-542915_1     :: CELL_DIFF ::          None , ATOM_DIFF ::          None
Al_mp-134_1           :: CELL_DIFF ::          None , ATOM_DIFF ::          None
Cu_mp-30_1            :: CELL_DIFF ::          None , ATOM_DIFF ::          None
Cu3Pd_mp-580357_1     :: CELL_DIFF ::          None , ATOM_DIFF ::          None
CuPd3_mp-1184119_1    :: CELL_DIFF ::          None , ATOM_DIFF ::          None
CuPd_mp-1018029_1     :: CELL_DIFF ::          None , ATOM_DIFF ::          None
Pd_mp-2_1             :: CELL_DIFF ::          None , ATOM_DIFF ::          None
PdPt_mp-1219908_1     :: CELL_DIFF ::          None , ATOM_DIFF ::          None
Pt_mp-126_1           :: CELL_DIFF ::          None , ATOM_DIFF ::          None
TiAl_mp-1953_1        :: CELL_DIFF ::          None , ATOM_DIFF ::          None
Ti_mp-72_1            :: CELL_DIFF ::          None , ATOM_DIFF ::          None
```
