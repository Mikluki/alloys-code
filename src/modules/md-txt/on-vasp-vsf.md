# VASP conventions

## Iterative Steps

- Electronic step = convergence iteration (inner loop).

- Ionic step = actual MD timestep/frame (outer loop → what you analyze).

### OUTCAR lines

From the VASP forum:

- `aborting loop because EDIFF is reached`
  → **electronic** minimization converged (SCF loop done).

- `reached required accuracy - stopping structural energy minimisation`
  → **ionic** minimization converged (IBRION=1, geometry step done).

So:
`EDIFF` → _electrons_
`EDIFFG` → _ions (geometry step)_

---

For any OUTCAR:

- **Ionic step count** = number of `POSITION ... TOTAL-FORCE` blocks (or equivalently, total atom-lines / NIONS).
- **Never** use `aborting loop because EDIFF is reached` to count MD frames; that’s an SCF thing.

> Keep in mind that whether or not a calculation is properly converged still depends on the situation (for example, the default EDIFF might be too large for accurate force/phonon calculations).

## Force Block (for ionic step)

```
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
     13.62318      9.68437      9.19945         0.450387     -0.557002      1.292850
```

Each row corresponds to **one atom** in the order given by POSCAR.

Columns 1–3:
**Atomic position** in **Cartesian coordinates (Å)**

```
x   y   z   (in Å)
```

Columns 4–6:
**Force vector on that atom** in **electron-volts per ångström (eV/Å)**

```
Fx   Fy   Fz   (in eV/Å)
```

The header

```
POSITION                                       TOTAL-FORCE (eV/Angst)
```

literally means:

- Left side = Cartesian coordinates
- Right side = Cartesian force components
- Units: positions in ångström; forces in eV per ångström

Positive force means “force acting in +axis direction” according to VASP’s coordinate system.

So your example row reads as:

- Atom at (13.62318 Å, 9.68437 Å, 9.19945 Å)
- Force = ( +0.450387, −0.557002, +1.292850 ) eV/Å

## Stress Block

- **Stress** → exists **per cell** (6 independent tensor components for the whole periodic box).
  Because stress is defined as **force per area acting on the boundaries of the unit cell**.
  The entire simulation box has one stress tensor at each ionic step.

```
FORCE on cell = -STRESS in cart. coord.  units (eV):

Direction    XX   YY   ZZ   XY   YZ   ZX

... contributions ...

Total     -54.31571   -97.02729   -30.52772     3.90898     6.44057    12.84782
in kB     -12.39167   -22.13595    -6.96464     0.89180     1.46936     2.93112
```

Interpretation:

- Line `Total` = the **stress tensor components** in eV/Å³-like internal units.
- Line `in kB` = **same stress tensor in kilobar**, the conventional unit.

Order is:

| Component | Meaning                      |
| --------- | ---------------------------- |
| XX        | σ<sub>xx</sub> normal stress |
| YY        | σ<sub>yy</sub> normal stress |
| ZZ        | σ<sub>zz</sub> normal stress |
| XY        | σ<sub>xy</sub> shear stress  |
| YZ        | σ<sub>yz</sub> shear stress  |
| ZX        | σ<sub>zx</sub> shear stress  |

This is **Voigt notation**.

Every ionic step gives you exactly **one** such tensor.

### Decomposition Lines (Ewald, Hartree, etc.)

VASP breaks the total stress into physical contributions:

- Alpha Z (nuclear)
- Ewald electrostatics
- Hartree energy
- Exchange correlation
- PAW augmentation
- Kinetic
- etc.

These are just diagnostic.
**You only care about the `Total` line (or `in kB`).**

# VSF Conventions

- for logging use pattern:

  ```python
  import logging
  LOGGER = logging.getLogger(__name__)
  ```

  > Logging is configured already

- we support only Path objects, no str, no nothing
- base funciton works with a given outcar path

Contract pattern:

- we store everything in a container dataclass that can be directly passed around. this is a contract

## stress array

```
stress_array (npt.NDArray): The 6-component stress tensor in Voigt notation,
    ordered as [σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy].
    Units are typically eV/Å³ if obtained from ASE or DFT calculations.

```
