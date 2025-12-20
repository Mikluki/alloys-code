## **Formation Energy: Definition**

Formation energy answers: "How much energy is released (or required) when you assemble a compound from its pure elemental reference states?"

$$
E_{\text{formation, per atom}} = \frac{E_{\text{composition}} - \sum_{i} N_i E_i}{\sum_i N_i}
$$

**Key requirements:**
1. All energies at **same conditions** (typically 0K, fully relaxed)
2. Reference states are **stable phases** (e.g., FCC Cu, BCC Fe, graphite C)
3. Structures at **equilibrium** (forces minimized, not moving)

### **Why Your Liquid Configs Don't Work for This:**

Your `energy.json` contains:
- **E0 from liquid snapshot at 300K** - NOT at 0K
- **Non-equilibrium configuration** - atoms in motion, not relaxed
- **Specific instantaneous arrangement** - not the thermodynamic ground state

If you tried to calculate formation energy:
$$E_{\text{formation}} = E_{\text{liquid snapshot}} - N \cdot E_{\text{bulk reference}}$$

This would be nonsense because:
- Left side: random liquid configuration at 300K
- Right side: equilibrium bulk at 0K
- You're mixing temperatures and thermodynamic states

### **What Your E0 Actually Represents:**

E0 is the **electronic energy of that specific atomic arrangement** at sigma→0.

It's useful for:
- **GNN benchmarking**: "Can the GNN predict VASP's energy for this geometry?"
- **Relative comparisons**: "Which liquid snapshot is lower/higher energy?"
- **ML training**: Direct structure → energy mapping

It's NOT useful for:
- Formation energy calculations (wrong thermodynamic state)
- Phase stability analysis (not equilibrium)
- Comparison to experimental data (experiments measure equilibrium properties)

### **What You Should Call It:**

In your GNN work, call it:
- "VASP reference energy" ✓
- "DFT energy for liquid configuration" ✓
- "Target energy for ML" ✓

Don't call it:
- "Formation energy" ✗
- "Ground state energy" ✗
- "Equilibrium energy" ✗

These are snapshot energies from MD - they're perfectly valid for ML benchmarking (structure in → energy out), but they exist in a different thermodynamic context than formation energies.

## Energy in OUTCAR

Short answer: you’re right to avoid the **free energy (TOTEN)** for 0 K formation energies. The usual convention is to use an energy **without the electronic-entropy term**. In practice:

- For **periodic solids** (most bulk/slab/surface calcs), use **`energy(sigma->0)`** from the last “FREE ENERGIE OF THE ION-ELECTRON SYSTEM” block in `OUTCAR`. This is VASP’s extrapolation to zero smearing (0 K) and is what toolchains like pymatgen treat as the final total energy. ([pymatgen][1]) For **atoms and small molecules**, VASP recommends **`energy without entropy`**, because the finite-T smearing can create unphysical electronic entropy for open-shell species; the VASP tutorial calls this out explicitly. ([vasp.at][2])

### What each line in OUTCAR means (last ionic step)

```
free  energy   TOTEN      = F = E − T*S_elec        (includes −T*S term; don’t use for 0 K formation energies)
energy  without entropy   = E = F + T*S_elec        (no entropy; at your chosen SIGMA)
energy(sigma->0)          ≈ E at SIGMA→0            (no entropy; extrapolated to 0 K)
```

Authoritative sources and libraries (VASP tutorials/wiki, pymatgen) use `energy(sigma->0)` as the **ground-state energy** for solids. ([vasp.at][3])

### Formation energy convention (0 K, DFT)

Formation energy is then computed from these 0 K total energies:
[
\Delta E_f = E_{\text{compound}} - \sum_i n_i \mu_i
]
using consistent reference energies ( \mu_i ) for elements (same functional, same U, etc.). Large databases (e.g., Materials Project) then apply fitted reference and mixing corrections on top of those totals to compare across chemistries; if you want MP-compatible values, you’d add those corrections after parsing. ([docs.materialsproject.org][4])

### So should we parse `OUTCAR`?

Yes. For your stability/convex-hull work:

- **Parse `energy(sigma->0)`** for periodic systems (bulk/slabs).
- **Parse `energy without entropy`** for isolated atoms/molecules used as references.
- **Do not** use `TOTEN` unless you explicitly need the electronic **free energy** at finite electronic temperature (rare for phase stability at 0 K). Reports and user threads consistently confirm that `TOTEN` includes the (-T S\_{\text{elec}}) term. ([GitHub][5])

[1]: https://pymatgen.org/pymatgen.io.vasp.html?utm_source=chatgpt.com "pymatgen.io.vasp package"
[2]: https://vasp.at/wiki/O_atom?utm_source=chatgpt.com "O atom - VASP Wiki"
[3]: https://www.vasp.at/tutorials/latest/molecules/part1/?utm_source=chatgpt.com "Part 1: Introduction to VASP"
[4]: https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/thermodynamic-stability?utm_source=chatgpt.com "Energy Corrections | Materials Project Documentation"
[5]: https://github.com/deepmodeling/dpdata/discussions/411?utm_source=chatgpt.com "what energy dpdata reads in the VASP OUTCAR file #411"

## MD && OSZICAR

### **OSZICAR Energy Types Explained:**

```
FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
---------------------------------------------------
free  energy   TOTEN  =      -213.36895200 eV        ← Includes -TS term (finite T)

energy  without entropy=     -213.00596576  energy(sigma->0) =     -213.18745888
       ↑ THIS ONE                                     ↑ Extrapolated to T=0K
```

POSCARs typically store `energy without entropy` (the internal energy E, not the free energy F).

- **TOTEN (F)** = E - TS (what the MD simulation minimizes)
- **energy without entropy (E)** = Internal energy (what you want for single structures)
- **energy(sigma->0)** = Smeared extrapolation (for metals with partial occupancies)

---

### **Mapping OSZICAR ↔ XDATCAR ↔ OUTCAR:**

```
XDATCAR                    OSZICAR                   OUTCAR
----------------------------------------
configuration= 1    →     Step 1 F=...    →    POSITION iteration 1
configuration= 2    →     Step 2 F=...    →    POSITION iteration 2
configuration= 100  →     Step 100 F=...  →    POSITION iteration 100
```

**They're synchronized by ionic step number.** Each `configuration=N` in XDATCAR corresponds to line N in `grep "F=" OSZICAR`.
