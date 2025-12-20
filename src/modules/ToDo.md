## Liquid MD

- Study the liquid silicon example on the wiki
  - Do the same for your elements (copy & adapt)

- Suggested calculation cell size: `n_atoms = 32` or `64`

- Target elements: `[Al, Na, Au, Cu, Ni]`

### Task: Sampling liquid metal configurations via MD in VASP

**Goal:**
Generate representative atomic configurations of liquid-phase metals for the listed elements using ab initio molecular dynamics (AIMD) simulations in VASP.

**Requirements:**

1. **System Setup**
   - Build supercells containing approximately 32–64 atoms of each target element (`Al`, `Na`, `Au`, `Cu`, `Ni`).
   - Start from appropriate crystalline structures (FCC, BCC, etc.) and randomize atomic positions slightly to initiate melting.

2. **MD Simulation**
   - Run ab initio molecular dynamics (AIMD) simulations at temperatures above each element’s melting point to reach the liquid state.
   - Use settings analogous to the liquid silicon example on the VASP wiki.

3. **Trajectory Sampling**
   - From each MD trajectory, sample **10 independent trajectories**, each producing **100 configurations** (snapshots of atomic positions).
   - Ensure sufficient equilibration time before sampling starts to guarantee liquid-like behavior.

4. **Output**
   - Save all sampled configurations as separate `POSCAR`/`CONTCAR` files.
   - Record simulation parameters (temperature, timestep, number of atoms, etc.) for reproducibility.

## Presentation

