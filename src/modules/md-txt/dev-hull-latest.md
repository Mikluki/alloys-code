# hull setup

## 1. What the VASP setup is actually doing

Given your INCAR:

- `NSW = 99`, `IBRION = 1`, `ISIF = 2`
  → Relax **internal coordinates only**, fixed cell (shape + volume fixed), up to 99 ionic steps.

- So for each volume-scaled structure (0.8 … 1.2), you get:
  - Cell fixed at that volume scaling factor
  - Atoms relaxed inside the cell
  - Final total energy `E_VASP`
  - Final stress tensor `σ_VASP` (from OUTCAR / vasprun)
  - Thus a hydrostatic-ish pressure:
    [
    P_{\text{VASP}} = -\tfrac{1}{3} \mathrm{Tr}(\sigma_{\text{VASP}})
    ]

- Other relevant bits:
  - `PREC = ACCURATE`, `ENCUT = 520` → plane-wave basis converged enough to serve as “truth”.
  - `ISPIN = 2`, `LASPH = TRUE`, GGA, smearing etc. define the electronic ground state you’re calling “reference physics”.
  - k-point settings presumably fixed across all volumes (even if via KPOINTS file instead of `KSPACING`).

So conceptually: for each MP structure on (or near) the hull, you’ve constructed an **E(V) and P(V)** curve with internal relaxation but fixed cell.

---

## 2. What “identical setups” should mean for GNN vs VASP

You can’t literally make GNN “use the same INCAR”, but you _can_ enforce:

1. **Identical geometries**:
   - Same cell vectors (volume scaling factors from the list)
   - Same atomic numbers
   - Same atomic positions (fractional or Cartesian)

2. **Same thermodynamic point**:
   - Both are zero-temperature, athermal models (DFT with smearing is effectively 0 K here; GNN is by construction 0 K).

3. **Same protocol**:
   - Option A: Compare on **VASP-relaxed geometries** for each volume.
   - Option B: Also explore **GNN-relaxed geometries** at each volume to see structural drift.

I’d strongly recommend:

- **Primary comparison**: _GNN on VASP geometries_ (cleanest, no “double relaxation” confounder).
- **Secondary comparison**: GNN-only relaxations as a robustness test.

---

## 3. Concrete experiment definition

### 3.1. Dataset construction (VASP side)

For each Material Project structure (i \in {1, \dots, N}):

- You already have 21 scaled volumes:
  [
  s \in S = {0.80, 0.82, \dots, 1.20}
  ]
- For each scale (s):
  1. Start from the MP equilibrium cell ( \mathbf{h}\_i ) and positions.
  2. Scale the **cell volume** to ( V = s \cdot V_0 )
     (if you did isotropic scaling, that’s ( \mathbf{h} \leftarrow s^{1/3} \mathbf{h} ); just stick to whatever you actually used).
  3. Keep fractional coordinates fixed as initial guess.
  4. Run VASP with your INCAR (ISIF=2, etc.) to relax internal coordinates.
  5. Extract after relaxation:
     - ( E\_{\text{VASP}}(i,s) ) — total energy
     - ( \sigma\_{\text{VASP}}(i,s) ) — stress tensor
     - ( \mathbf{R}\_{\text{VASP}}(i,s) ) — relaxed atomic positions
     - Volume ( V(i,s) )

Store:

- Energy _per atom_: ( e*{\text{VASP}}(i,s) = E*{\text{VASP}}(i,s) / N\_{\text{atoms},i} )
- Pressure:
  [
  P_{\text{VASP}}(i,s) = -\frac{1}{3} \mathrm{Tr}(\sigma_{\text{VASP}}(i,s))
  ]
- Optionally deviatoric stress:
  (\sigma^{\text{dev}} = \sigma - \tfrac{1}{3}\mathrm{Tr}(\sigma)\mathbf{I})

---

### 3.2. GNN evaluation protocol

Assume you have a pre-trained potential (MACE / SevenNet / whatever).

For each **VASP-relaxed structure** ((i,s)):

1. Feed **the exact same cell** and **positions (\mathbf{R}\_{\text{VASP}}(i,s))** into the GNN.

2. Compute:
   - ( E\_{\text{GNN}}(i,s) )
   - Forces ( \mathbf{F}\_{\text{GNN}}(i,s) )
   - Stress tensor ( \sigma\_{\text{GNN}}(i,s) )
     (many frameworks can give virial stress directly; if not, compute from virial formula)

3. Define:
   - Energy per atom:
     [
     e_{\text{GNN}}(i,s) = E_{\text{GNN}}(i,s) / N_{\text{atoms},i}
     ]
   - Pressure:
     [
     P_{\text{GNN}}(i,s) = -\tfrac{1}{3}\mathrm{Tr}(\sigma_{\text{GNN}}(i,s))
     ]

4. (Optional extra experiment) GNN-relaxation per volume:
   - Start from the _same_ scaled cell and initial positions as VASP did.
   - Optimize atoms using GNN forces with fixed cell until (|F| < F\_\text{thresh}).
   - Then record ( E*{\text{GNN}}^{\text{relax}}(i,s), P*{\text{GNN}}^{\text{relax}}(i,s) ), etc.

---

## 4. Quantities and plots you actually compare

You mentioned “pressure normalised by volume vs volume coefficient” – that’s basically a rescaled equation of state.

For each structure (i):

1. Define normalized volume:

   $$
   \tilde{V}(i,s) = \frac{V(i,s)}{V_0(i)}
   $$

   where (V_0(i)) is the MP equilibrium volume (your (s=1) case).

2. For each method (M \in {\text{VASP}, \text{GNN}}):
   - Plot:
     - Energy vs normalized volume:
       $$
       e_M(i,s) - e_M(i, s=1)
       \quad \text{vs} \quad
       \tilde{V}(i,s)
       $$
     - Pressure vs normalized volume:
       $$
       P_M(i,s) \quad \text{vs} \quad \tilde{V}(i,s)
       $$
     - If you really want “pressure normalized by volume”, that’s (P_M(i,s) / \tilde{V}(i,s)) or (P_M(i,s)V(i,s)); just define it clearly once and stick to it.

3. Fit equation of state (Birch–Murnaghan or similar) separately to
   - VASP (E(V))
   - GNN (E(V))

   and compare:
   - Equilibrium volume: (V*{0,\text{GNN}}) vs (V*{0,\text{VASP}})
   - Bulk modulus: (B*{0,\text{GNN}}) vs (B*{0,\text{VASP}})
   - Maybe also derivative (B_0') if you care about curvature.

4. Error metrics across all (i,s):
   - Pressure error:

     $$
     \Delta P(i,s) = P_{\text{GNN}}(i,s) - P_{\text{VASP}}(i,s)
     $$

     and report RMSE / MAE over:
     - all volumes,
     - or separately for “small strain” (|s-1| \le 0.04) vs “large strain”.

   - Energy error:

     $$
     \Delta e(i,s) = \big(e*{\text{GNN}}(i,s) - e*{\text{GNN}}(i,1)\big) -
     \big(e*{\text{VASP}}(i,s) - e*{\text{VASP}}(i,1)\big)
     $$

     so you’re comparing _shape_ of the curve, not absolute offset.

   - Optional: norm of force disagreement on VASP-relaxed geometries as a function of strain to see how “itchy” the GNN feels about VASP’s structures away from equilibrium.

This will show exactly the “very different curves from VASP” behavior you expect, but in a quantifiable way.

