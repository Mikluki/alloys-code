# Experiment setup


---

# Details

## ASE schemes

Because different algorithms trade off:

- **Correctness of ensemble** (does it sample the exact canonical or NPT distribution?),
- **Robustness / stability** (does it blow up if you’re slightly sloppy?),
- **Dynamical realism** (how badly does it distort time correlations).

Quick character sketches of the ones you mentioned:

- `NVTBerendsen`
  - Deterministic weak-coupling thermostat.
  - Simple and robust, but **does not generate a strictly correct canonical ensemble**; fluctuations are suppressed.
  - Good for equilibration, a bit dodgy for precise thermodynamics.

- `Inhomogeneous_NPTBerendsen`
  - Berendsen thermostat + inhomogeneous barostat.
  - Same story, but also adjusts the cell → NPT-ish, not what VASP used here.

- `NPT` (ASE)
  - Parrinello–Rahman-style barostat + Nosé–Hoover chain thermostat. ([databases.fysik.dtu.dk][4])
  - A proper NPT method when correctly tuned.
  - But it explicitly evolves the cell → not a match to your fixed-cell VASP run unless you hack it.

Other relevant ASE options (you didn’t list, but they’re important):

- `Langevin`
  - Stochastic thermostat: friction + random force. ASE explicitly recommends it as a good NVT sampler. ([wiki.fysik.dtu.dk][5])
  - Pros: simple, numerically robust, gives correct NVT ensemble if parameters are sensible.
  - Cons: trajectory is stochastic and dynamics are slightly overdamped depending on friction.

- `NoseHooverChainNVT`
  - Pure NVT Nosé–Hoover chain thermostat in ASE, implemented carefully. ([databases.fysik.dtu.dk][4])
  - This is the **closest conceptual analogue to VASP `MDALGO = 2`**.
  - Needs a damping timescale parameter `tdamp`.

So: many schemes exist because MD has _two_ tasks:

1. Integrate the equations of motion stably.
2. Sample a desired statistical ensemble.

Different algorithms emphasize different aspects of that tradeoff.

## Metrics to compare against VASP (including collapse behaviour)

You already like RDF and MSD — good instincts. Let’s make the metric set explicit and structured.

### “Equilibrium ensemble” metrics

These are averaged over time windows where the simulations are still healthy.

Use the same analysis window length for VASP and GNN, e.g. last 20 ps of each run (or up to collapse).

**(a) Temperature stats**

From kinetic energy:

- (\langle T \rangle)
- standard deviation (\sigma_T)
- maybe distribution (P(T)) over time (histogram).

Compare:

- mean close to 1400 K?
- are fluctuations of similar magnitude to VASP?

**(b) Potential energy distribution**

Track (E\_\text{pot}(t)):

- mean and variance,
- autocorrelation time of energy.

This tells you whether the GNN explores similar parts of the energy landscape.

**(c) RDFs (g\_{\alpha\beta}(r))**

Per pair type and maybe total:

- (g*{\text{Al–Al}}(r)), (g*{\text{Al–Cu}}(r)), (g*{\text{Al–Ni}}(r)), (g*{\text{Cu–Ni}}(r)), etc.

Compute with ASE `Analysis` , for both trajectories:

- Compare positions and heights of peaks,
- Integrate first peak to get coordination numbers.

If GNN RDFs are close to VASP’s, that’s strong evidence structural statistics are reproduced.

**(d) MSD and diffusion**

For each species (\alpha):

$$
\text{MSD}_\alpha(t) = \left\langle \left|\mathbf{r}_i(t_0 + t) - \mathbf{r}*i(t_0)\right|^2 \right\rangle*{i\in\alpha, t_0}
$$

Fit the linear regime at long times:

$$
\text{MSD}*\alpha(t) \approx 6 D*\alpha t
$$

Compare diffusion constants (D*\alpha^\text{VASP}) vs (D*\alpha^\text{GNN}).

This is especially relevant at 1400 K where you likely have molten or highly diffusive behaviour.

**(e) Velocity distribution**

Optional but cheap: check that the velocity components are Gaussian with variance corresponding to 1400 K:

- Extract velocity histograms for each species,
- Fit a Maxwell–Boltzmann or just compare second moment.

If this matches, your thermostat is behaving.

---

### “Stability / collapse” metrics

You already anticipate the ML potential may run fine for a while, then suddenly go to hell. Let’s treat that as a measurable quantity.

Define **stability indicators** you monitor every N steps:

- **Max force per atom**:
  (F\_\text{max}(t) = \max_i |\mathbf{F}\_i(t)|)
- **Minimum interatomic distance**:
  (r*\text{min}(t) = \min*{i\neq j} |\mathbf{r}\_i(t) - \mathbf{r}\_j(t)|)
- **Energy spikes**: sudden jumps (|E(t+\Delta t) - E(t)|) beyond some threshold.

Then define a “collapse event” if any of these conditions is met:

- (F*\text{max}(t) > F*\text{threshold}) (e.g. > 50–100 eV/Å, pick based on your VASP ranges).
- (r*\text{min}(t) < r*\text{cut}) (e.g. atoms practically overlapping).
- Potential energy diverges or grows monotonically without bound.

For each seed:

- measure **time to collapse** (t\_\text{fail}) (or `None` if it survives entire run),
- summarise as distribution over seeds.

This gives you two nice, quantitative statements:

1. “Under these conditions, MACE MD is stable over X ps in Y/Z seeds.”
2. “When stable, ensemble properties (RDF, MSD, etc.) match VASP with such-and-such deviation.”

That’s much more scientific than “it kinda runs and sometimes blows up”.

---

### Direct trajectory-level comparison (optional, but cool)

If you really want to go nerd maximalism:

- Start GNN MD from a VASP frame and run **short** (e.g. 0.5–1 ps) with same initial velocities.
- Compare how quickly the trajectories decorrelate:
  - RMSD vs time between VASP and GNN positions,
  - overlap of velocity directions.

But note: even two _DFT_ runs with slightly different thermostats will decorrelate exponentially fast; dynamical equivalence is about ensemble properties, not matching individual atom paths. So I’d treat this as a curiosity, not a core metric.
