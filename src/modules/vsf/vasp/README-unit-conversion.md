# VASP vs ASE stresses

How do I compare vasp stresses and ase stresses for the same poscar?

For MP Al4

Stresses from ASE:

```
"MACE_MPA_0": {
  "stress_array": [
    8.138729401275166e-05,
    8.138729401275289e-05,
    8.138729401268441e-05,
    -7.991600449270171e-17,
    -1.0166771151651086e-17,
    7.970158183997435e-16
  ]
}
```

Stresses from OUTCAR

```
  Total      -0.08544    -0.08544    -0.08544    -0.00000     0.00000     0.00000
  in kB      -2.07760    -2.07760    -2.07760    -0.00000     0.00000     0.00000
  external pressure =       -2.08 kB  Pullay stress =        0.00 kB
```

## Direct calcs

You want to compare:

- VASP `OUTCAR` stress:

```text
Total      -0.08544    -0.08544    -0.08544    -0.00000     0.00000     0.00000
in kB      -2.07760    -2.07760    -2.07760    -0.00000     0.00000     0.00000
```

- ASE / MACE stress (eV/Å³):

```json
"stress_array": [
  8.138729401275166e-05,
  8.138729401275289e-05,
  8.138729401268441e-05,
  -7.99e-17,
  -1.02e-17,
  7.97e-16
]
```

These are **the same physical quantity**, but:

1. **Different units**
2. **Different sign convention**
3. **Different shear order**

---

### 1. Units and sign: VASP → ASE

VASP’s `in kB` line is in **kilobar**. ASE uses **eV/Å³**.

Conversion:

- (1~\text{eV/Å}^3 \approx 160.21766208~\text{GPa} \approx 1602.1766208~\text{kbar}) ([wiki.fysik.dtu.dk][1])
- So
  [
  1~\text{kbar} = \frac{1}{1602.1766208}~\text{eV/Å}^3 \approx 6.24\times 10^{-4}~\text{eV/Å}^3
  ]

VASP and ASE also use **opposite stress sign conventions**. VASP’s kB stresses come out with the opposite sign compared to ASE’s internal convention, and ASE explicitly multiplies by `-1` when reading `OUTCAR`. ([GitHub][2])

ASE’s own VASP reader does:

```python
## from ase.calculators.vasp.vasp.read_stress
if ' in kB  ' in line:
    stress = -np.array([float(a) for a in line.split()[2:]])  # flip sign
    stress = stress[[0, 1, 2, 4, 5, 3]] * 1e-1 * ase.units.GPa
```

So mathematically:

[
\sigma_{\text{ASE}}(\text{eV/Å}^3) = -,\frac{\sigma_{\text{VASP}}(\text{kbar})}{1602.1766208}
]

---

### 2. Shear component order

VASP’s `in kB` line is:

```text
sxx  syy  szz  sxy  syz  szx
```

ASE’s `get_stress(voigt=True)` returns stress in **Voigt order**:

```text
[ sxx, syy, szz, syz, sxz, sxy ]
```

So the permutation is:

```python
## VASP: [0, 1, 2, 3, 4, 5] = [xx, yy, zz, xy, yz, zx]
## ASE:  [xx, yy, zz, yz, xz, xy] = indices [0, 1, 2, 4, 5, 3]
perm = [0, 1, 2, 4, 5, 3]
```

Exactly what you see in ASE’s code. ([wiki.fysik.dtu.dk][3])

---

### 3. Putting it together: direct conversion snippet

If you’ve already parsed the `in kB` line into a list/array:

```python
import numpy as np

def vasp_kb_to_ase_evA3(stress_kb):
    """
    stress_kb: array-like, 6 components from 'in kB' line
               [sxx, syy, szz, sxy, syz, szx] in kbar (VASP)
    returns:   np.ndarray, 6 components in ASE Voigt order
               [sxx, syy, szz, syz, sxz, sxy] in eV/Å³ (ASE convention)
    """
    stress_kb = np.array(stress_kb, dtype=float)
    perm = [0, 1, 2, 4, 5, 3]  # VASP -> ASE Voigt
    kb_to_eVA3 = 1.0 / 1602.1766208

    return -stress_kb[perm] * kb_to_eVA3
```

Use on your example:

```python
vasp_kb = [-2.07760, -2.07760, -2.07760, -0.0, 0.0, 0.0]
ase_from_vasp = vasp_kb_to_ase_evA3(vasp_kb)
print(ase_from_vasp)
## ~ [0.001297, 0.001297, 0.001297, 0., 0., 0.]  eV/Å³
```

Your MACE stress is:

```python
mace = 8.1387e-05  # eV/Å³
```

So:

```python
ratio = mace / ase_from_vasp[0]  # ~ 0.063
```

Meaning MACE_MPA_0 is predicting a hydrostatic stress about **16× smaller** in magnitude than the VASP reference here — totally plausible model error, not a unit bug.

---

### 4. “Idiot-proof” way: let ASE do the parsing

You can also sidestep all this by letting ASE read the `OUTCAR`:

```python
from ase.io import read

atoms_vasp = read("OUTCAR")  # ASE parses stresses and flips sign etc.
vasp_stress_ase = atoms_vasp.get_stress(voigt=True)  # eV/Å³, ASE convention

## Now directly comparable to your MACE stress:
atoms = structure.to_ase_atoms()
atoms.calc = mace_calc
mace_stress = atoms.get_stress(voigt=True)

print("VASP (ASE):", vasp_stress_ase)
print("MACE     :", mace_stress)
print("Δstress  :", mace_stress - vasp_stress_ase)
```

That’s the cleanest way to be sure your DFT vs GNN comparisons are “same POSCAR, same units, same sign, same ordering” and you’re only measuring physics mismatch, not convention chaos.

[1]: https://wiki.fysik.dtu.dk/ase/ase/units.html?utm_source=chatgpt.com "Units — ASE documentation"
[2]: https://github.com/libAtoms/QUIP/issues/227?utm_source=chatgpt.com "The sign of stress and virial used in gap_fit training #227"
[3]: https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/vasp/vasp.html?utm_source=chatgpt.com "ase.calculators.vasp.vasp — ASE documentation"

## Al4 ase interface

Input:

```
VASP (ASE): [ 0.00129674  0.00129674  0.00129674 -0.         -0.          0.        ]
MACE      : [ 8.13872940e-05  8.13872940e-05  8.13872940e-05 -7.99160045e-17
 -1.01667712e-17  7.97015818e-16]

Δstress   : [-1.21534864e-03 -1.21534864e-03 -1.21534864e-03 -7.99160045e-17
 -1.01667712e-17  7.97015818e-16]
```


Yeah, that *looks* weird at first glance, but at this point it’s not a units/sign bug anymore — it’s just the model disagreeing with VASP.

Let’s decode what you’ve got:

```text
VASP (ASE): [ 0.00129674  0.00129674  0.00129674 -0.         -0.          0.        ]
MACE      : [ 8.1387e-05  8.1387e-05  8.1387e-05 ~0 ~0 ~0 ]

Δstress   : [-1.2153e-03 -1.2153e-03 -1.2153e-03 ...]
```

### 1. Check the magnitude in “human” units

Convert to GPa:

* VASP:
  (0.00129674 ,\text{eV/Å}^3 \times 160.22 \approx 0.208 ,\text{GPa} \approx 2.08 ,\text{kbar})
  (matches your `in kB -2.0776` once you account for sign convention)
* MACE:
  (8.1387\times10^{-5} ,\text{eV/Å}^3 \times 160.22 \approx 0.013 ,\text{GPa})
* Difference:
  (1.2153\times10^{-3} ,\text{eV/Å}^3 \times 160.22 \approx 0.195 ,\text{GPa})

So MACE is underestimating the hydrostatic stress by ~0.2 GPa.

For context: bulk modulus of Al is ~70 GPa. So the “equivalent” strain error is roughly

[
\varepsilon \sim \frac{0.2}{70} \approx 0.003 \quad \text{(0.3% strain)}
]

Not catastrophic, but definitely non-zero.

### 2. Why this happens (and why it’s not a bug in your interface)

You did the right thing:

* Same POSCAR
* VASP stress parsed via ASE (`read("OUTCAR")`) → ASE units + convention
* MACE stress via `atoms.calc = mace_calc; atoms.get_stress(voigt=True)`

Those *should* be directly comparable — and they now are.

The remaining discrepancy is about **physics / training**, not **plumbing**:

1. **Many MLIP / GNN models are primarily trained on energies + forces.**
   Stresses are often:

   * not included at all, or
   * included with a much lower weight.

   In that case, stresses are effectively “second derivatives” inferred from the learned potential, and can be noticeably noisier / biased.

2. **Even when stresses are in the training set, they’re often a harder target.**
   Small absolute differences in energy per atom can translate into noticeable stress differences, especially for small cells like Al4.

3. **You’re looking at a tiny toy cell (Al₄).**
   For such a small system, discretization of k-points, finite-size effects, etc. in the DFT reference can give a stress that’s not super “bulk-like,” while the model has probably seen mostly larger, more bulk-ish configurations.

### 3. How to sanity-check that it’s *really* model error

You already did the main check (VASP via ASE vs MACE via ASE). A few extra quick probes if you want to convince yourself:

* **Scale the cell slightly and see how each responds.**

  For the same structure, try:

  ```python
  for scale in [0.99, 1.00, 1.01]:
      s_scaled = structure.copy()
      s_scaled.scale_lattice(s_scaled.volume * scale**3)

      atoms = s_scaled.to_ase_atoms()

      # VASP: read OUTCAR for each scaled structure (if you have them)
      # MACE:
      atoms.calc = mace_calc
      print(scale, atoms.get_stress(voigt=True))
  ```

  If the *trend* with scale is sensible and the model’s stresses track VASP qualitatively but with an offset, that’s classic “model approximates EOS but not perfectly” behavior.

* **Compare pressures instead of raw Voigt components.**

  ```python
  def pressure_from_stress(s):
      return - (s[0] + s[1] + s[2]) / 3.0

  p_vasp = pressure_from_stress(vasp_stress)
  p_mace = pressure_from_stress(mace_stress)
  ```

  That’s the quantity you probably care about for screening (and what you already started using as a metric).

### 4. What this means for your use case

So the weirdness is:

> “I nailed units and sign, and there’s still a ~0.2 GPa gap.”

Yup. That’s just the model not matching DFT stress perfectly. Typical strategies:

* If your application is **energy-/structure-driven**, and you only use stress as a *secondary* metric (e.g., to filter obviously strained configs), this might be perfectly fine.
* If you need **high-fidelity stresses** (e.g., for elastic constants, EOS fitting, accurate pressure control), you either:

  * need a model explicitly trained with stress terms heavily weighted, or
  * accept that this MACE flavor isn’t a drop-in replacement for VASP stresses.

In other words: your pipeline is doing the right thing; the numbers are telling you something real about the model, not your code.
