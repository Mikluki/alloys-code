## `converged_force`

**What it is:**
A boolean that says: _did this relaxation end with forces small enough that we’re willing to treat the final configuration as “relaxed” (for the fixed cell)?_

**What it compares explicitly:**

- `max_force` **from your parsed OUTCAR final ionic step**
  compared to
- a chosen force threshold `fmax_target_vasp` **in the same units** (eV/Å)

So:

- `converged_force = (max_force <= fmax_target_vasp)`

**Where does `fmax_target_vasp` come from?**
From your _acceptance criterion_, not from VASP magically. In VASP, the “force convergence” criterion is usually controlled by `EDIFFG` (often negative means forces), but you may or may not have a consistent `EDIFFG` across runs, and you’re already using a GNN criterion `fmax=0.02 eV/Å`. The point is: pick one threshold you’re comfortable with (often the same 0.02 eV/Å for symmetry), and label points that meet it.

**Why it matters:**
EOS curves are hypersensitive to “not actually relaxed” points: one unconverged point can create a fake kink in (E(V)) or a weird jump in (P(V)). This flag lets you:

- filter those points out of EOS fits,
- or keep them but mark them as low-confidence.

---

## `hit_nsw_limit`

**What it is:**
A boolean that says: _did the ionic relaxation stop because it ran out of allowed ionic steps, rather than because it converged?_

**What it compares explicitly:**

- `n_ionic_steps` **actually performed** (from OUTCAR)
  compared to
- `NSW` **the maximum you allowed** (here, 99)

So:

- `hit_nsw_limit = (n_ionic_steps >= NSW)`
  (or `== NSW` depending on whether you count from 0 or 1 — purely a bookkeeping convention.)

**Why it matters:**
If a run used up all 99 ionic steps, the most common interpretation is: it **didn’t converge within the budget**. It might still end with small forces (rare but possible), or it might be nowhere near converged. So this flag is an independent “smells like non-convergence” detector.

---

## How the two flags work together (the key idea)

They answer two different questions:

- **`converged_force`**: _Is the final state physically “relaxed enough” by your force standard?_
- **`hit_nsw_limit`**: _Did the optimizer likely stop for a “budget” reason?_

This gives you a simple 2×2 interpretation:

- **converged_force = True, hit_nsw_limit = False** → clean point (best case)
- **converged_force = False, hit_nsw_limit = True** → almost certainly bad point
- **converged_force = False, hit_nsw_limit = False** → stopped early for some other reason (also suspicious)
- **converged_force = True, hit_nsw_limit = True** → “weird but usable”: met force threshold exactly at the step limit; keep but mark

In practice, you’ll later do things like:

- EOS fits using only `converged_force == True` and `hit_nsw_limit == False`
- diagnostics plots that highlight the excluded points, so you know whether failures cluster at extreme volumes (they often do).

That’s the entire point: **stop pretending every OUTCAR is equally trustworthy** and bake that reality into the dataset.
