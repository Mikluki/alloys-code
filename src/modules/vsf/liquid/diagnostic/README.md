# Structural Decorrelation (VASP MD)

**Goal:** Convey physical intuition for decorrelation analysis and help users diagnose _bad_ vs _good_ sampling using diagnostic plots.

---

## Recommended parameters (TL;DR)

**Goal:** fast, physically sensible settings that pass the sanity plots and yield workable sampling.

### Defaults (good starting points)

| Material                    | frame_lag `k` |         δ (Å) | Rationale                                                                                           |
| --------------------------- | ------------: | ------------: | --------------------------------------------------------------------------------------------------- |
| **Na (liquid, T≈Tm+300 K)** |        **10** |      **0.05** | Nice variance, ACF 1/e ≈ 20–30 lags, good spread of kept frames.                                    |
| **Al (liquid)**             |        **10** | **0.06–0.08** | Lighter than Cu → slightly larger δ than 0.05 works better.                                         |
| **Cu (liquid)**             |        **10** |      **0.05** | Stable universal base; good variance without saturation.                                            |
| **Ni (liquid)**             |        **10** |      **0.05** | Similar to Cu; small tweaks ±0.01 as needed.                                                        |
| **Au (liquid)**             |        **60** |      **0.04** | Heavy atom → smaller steps and long correlations; needs larger `k` at similar δ to lift efficiency. |

### Quick rule of thumb

- Good range for liquid/metallic systems near melting: **k ≈ 10**, **δ ≈ 0.05 Å ± 0.02**.
- **Histogram mode** ~ **0.4–0.7** with visible width.
  - Mode > 0.75 → **δ too small** (saturating) → increase δ by **+0.01–0.02**.
  - Mode < 0.25 → **δ too large** (too sparse) → decrease δ by **−0.01–0.02**.

- **ACF 1/e crossing** within **15–40** lags (post-burn-in).
- **Target** `g_struct ≤ 15` and **sampling efficiency ≥ 5%** (post-burn-in).

### If efficiency is low (<5%)

1. **Increase `k`** (strongest lever to reduce correlation): try 20 → 40 → 60, keep δ fixed.
2. Touch δ only to fix **degenerate histograms** (hugging 0 or 1).

---

## Interpreting your Au results (what to settle on)

You tested **Au, δ=0.04 Å** with:

- `k=15`: overall ~**2.0%** (g_struct ~44–57)
- `k=40`: overall ~**2.6%** with big spread
  - seed4: g_struct **27.7** (OKish)
  - seed3: g_struct **39.0** (still high)
  - seed6: g_struct **60.4** (still very high)

**Conclusion:** Au is correlation-dominated at short gaps. δ=0.04 is fine; the knob that matters is **`k`**.

### Actionable pick for Au

- **Set `k = 60`, δ = 0.04 Å** as the **production default for Au**.
  Expect g_struct to drop into the 10–20 range on most runs and push efficiency toward/above **5%**. If any trajectory still shows a histogram hugging 0, nudge δ to **0.035–0.04** (don’t increase δ).

(If you prefer a single global setting across all elements, keep `k=10, δ=0.05` for Na/Al/Cu/Ni and **special-case Au** with `k=60, δ=0.04`.)

---

## Optional “auto-tune” recipe (one line of logic)

- Run a **pilot** on the first 1–2 ps per trajectory, sweep `k ∈ {10,20,40,60}` at a fixed δ, pick the smallest `k` that yields `g_struct ≤ 15` **and** a non-degenerate histogram.
- Keep those `k, δ` for the full run.

This keeps interpretation simple while making Au behave without hand-holding.

---

## 1) Naming + Core Definitions

- **`frame_lag (k)`** — number of frames between configurations compared when computing structural change. If frames are saved every `Δt_frame`, the physical interval is `k × Δt_frame`.
- **`displacement_threshold_Å (δ)`** — Ångström threshold for deciding whether an atom “moved”.
- **`moved_fraction f(t; k, δ)`** — fraction of atoms with minimum‑image displacement ≥ δ between frames `t` and `t−k`. Range: `[0, 1]`.
- **`equil_start_idx (t0)`** — first equilibrated frame (from potential energy via `pymbar.detect_equilibration`).
- **`g_energy`, `g_struct`** — statistical inefficiency (frames per independent sample) estimated from the energy and structural series, respectively.
- **`acf_lag (ℓ)`** — axis for the autocorrelation plot of the structural series (kept distinct from `frame_lag`).

> **Why “lag”?** In time‑series analysis, “lag” is the index offset between samples. Here, `frame_lag` sets the **comparison gap** for the structural observable. Larger `k` suppresses fast vibrations (low‑pass effect) and emphasizes slower reorganizations.

---

## 2) Physical Meaning of `frame_lag`

- **Small `k` (1–3):** compares nearby frames → sensitive to fast thermal vibrations; high time resolution; noisy, meaningless motion.
- **Moderate `k` (5–15):** suppresses vibrations; highlights local rearrangements; smoother signal.
- **Large `k` (≥20):** emphasizes slow collective motion (diffusion, cage breaking); smooth but may miss short‑lived events.

**Rule of thumb:** choose `k` so that `f(t; k, δ)` is neither flat (0 or 1) nor chaotic. Aim for a wide, stable band in `[0, 1]` after burn‑in.

---

## 3) Workflow (Production Logic)

1. **Burn‑in detector (Filter #1):** potential‑energy time series → `t0` via `pymbar.detect_equilibration()`.
2. **Structural decorrelation (Filter #2):** compute `f(t; k, δ)` and subsample post‑burn‑in via `alchemlyb.preprocessing.subsampling.statistical_inefficiency()`; report `g_struct`.
3. **Sanity checks (Plots):** time series with kept frames, histogram (post‑burn‑in), and ACF of `f(t; k, δ)`.

**Summary:** energy finds **equilibration**; the structural series performs **thinning**; plots are **diagnostics** of independence.

---

## 4) How to Read the Diagnostic Plots

### A) Structural Time Series (left panel)

- **What to see:** blue line stabilizes after red `t0`; orange kept frames span equilibrated region; visible amplitude of motion.
- **Flags:** flat near 1 or 0 → degenerate metric; dense clustering of orange points → correlation not removed.

### B) Histogram (middle panel)

- **What to see:** non‑zero width; no spike solely at 0 or 1.
- **Quick cue:** IQR ≳ **0.02** in fraction‑moved units is workable.

### C) ACF of `f(t; k, δ)` (right panel)

- **What to see:** exponential‑like decay; crosses 1/e within reasonable lag count.
- **Heuristics for ℓ₁ₑ (1/e crossing):**
  - **Good:** ℓ₁ₑ ≲ 15
  - **OK:** 15–40
  - **Poor:** > 40 or plateauing

### D) Statistical Inefficiency `g_struct`

- **Good:** `g_struct` ≲ 5 (keep ≥20%)
- **OK:** 5–15 (keep ~7–20%)
- **Poor:** >15 (keep <7%)

### E) Sampling Efficiency (actual kept)

- **Good:** ≥15%
- **OK:** 5–15%
- **Poor:** <5%

---

## 5) Example: Degenerate Metric (Bad Sampling)

**Parameters:** `frame_lag = 5`, `δ = 0.005 Å`

**Observed features:**

- Blue curve pinned at ~1 → every atom “moved” more than δ each frame gap.
- Histogram nearly δ‑function at 1 → no variance.
- ACF decays quickly but meaningless (flat signal).

**Physical interpretation:**

- Threshold too tight relative to thermal vibration amplitude → saturated metric.
- Observable fails to discriminate between random thermal motion and actual rearrangements.

**Fix:** increase `δ` (to ~0.05 Å) and/or increase `k` (10–20). This suppresses vibration noise and restores meaningful variation.

---

## 6) Example: Physically Reasonable Metric (Good Sampling)

**Parameters:** `frame_lag = 10`, `δ = 0.05 Å`

**Observed features:**

- Structural time series shows broad fluctuations around 0.6–0.7; orange points evenly spread.
- Histogram broad (IQR ≈ 0.1) with distinct peak; variance well‑defined.
- ACF crosses 1/e within ≈25 lags → reasonable correlation time.

**Interpretation:**

- Observable sensitive to meaningful rearrangements in the liquid structure.
- Sampling efficiency high; frames are largely independent.
- Metric consistent with fast atomic diffusion typical for molten metals ~300 K above melting point.

**Physical reasonableness:**

- For **liquid metals** near or above melting, atomic thermal displacement per 10 fs is O(0.05 Å). Using δ≈0.05 Å filters out ballistic vibrations while detecting cage changes and diffusive hops.
- Thus `(frame_lag=10, δ=0.05 Å)` provides a realistic balance for decorrelation in molten or disordered metallic systems.

---

## 7) Tuning Recipes

| Symptom           | Diagnosis                     | Adjustment              |
| ----------------- | ----------------------------- | ----------------------- |
| `f≈1` (flat)      | δ too small                   | Increase δ or k         |
| Flat `f≈0`        | δ too large                   | Decrease δ or k         |
| Noisy sawtooth    | k too small                   | Increase k              |
| ACF decays slowly | Structural correlation strong | Increase k or extend MD |
| Few frames kept   | High g                        | Tune δ/k or run longer  |

---

## 8) Minimal Math

Autocorrelation of observable `A(t)`:

$$
C(\tau) = \frac{\langle A(t)A(t+\tau)\rangle - \langle A \rangle^2}{\langle A^2 \rangle - \langle A \rangle^2}, \quad g = 1 + 2\sum_{\tau=1}^{\infty} C(\tau).
$$

`g` ≈ frames per independent sample; `N_eff = N/g`.

Fraction‑moved observable:

$$
f(t; k, \delta) = \frac{1}{N}\sum_i \Theta(|\mathbf r_i(t) - \mathbf r_i(t-k)|_{\text{MIC}} - \delta) \in [0,1].
$$

---

## 9) One‑page Checklist (Operator View)

- [ ] Confirm energies present; detect burn‑in (`t0`).
- [ ] Compute `f(t; k, δ)`; verify histogram has width.
- [ ] Time series stable post‑burn‑in.
- [ ] ACF crosses 1/e within 15–40 lags.
- [ ] `g_struct ≤ 15` (prefer 5–10).
- [ ] Sampling efficiency ≥ 5–15%.

---

### Short Glossary

- **Burn‑in (`t0`)** — equilibrium onset from energy.
- **Statistical inefficiency (`g`)** — frames per independent sample.
- **MIC** — minimum‑image convention.
- **ACF** — autocorrelation function of scalar time series.

---

## 10) Recommended δ by Material Class (with rationale)

**Purpose:** give starting δ values that align with physically reasonable motion scales; always validate via histogram/ACF.

| Class                                         | Motion scale (order-of-magnitude)                                                                         | Suggested δ (Å)              | Rationale                                                                                                                             |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Crystalline solids (near RT–moderate T)**   | Thermal RMS displacements from Debye–Waller/ADP; typical √⟨u²⟩ of O(10⁻³–10⁻¹) Å depending on T & element | **0.01–0.05**                | δ should exceed instantaneous vibrational jitter but stay < intersite motion; see Debye–Waller background and reported ADPs (Fe, Na). |
| **Amorphous / glassy**                        | Larger static disorder + thermal jitter than crystals                                                     | **0.02–0.08**                | Needs higher δ than crystals to avoid classifying local jitter as "moved".                                                            |
| **Liquid / molten metals (≈ Tₘ to Tₘ+300 K)** | Self‑diffusion D ~ 10⁻⁹–10⁻⁸ m²/s; over 10 fs, √(6Dt) ~ 0.03–0.1 Å                                        | **0.03–0.10** _(often 0.05)_ | Chooses δ around typical diffusive step over `k×Δt` (e.g., k=10 at 1 fs); captures cage breaks, filters vibrations.                   |
| **Molecular liquids / soft matter**           | Reorientation + translation; larger MSD on tens of fs                                                     | **0.05–0.2**                 | Heavier reorientation/translation warrants higher δ to avoid saturation.                                                              |
| **Ionic liquids / molten salts / oxides**     | Larger ionic steps                                                                                        | **0.05–0.2**                 | Higher δ tolerates larger displacements without immediate saturation.                                                                 |

**Notes:** (i) δ interacts with `frame_lag (k)` and output cadence `Δt_frame`; tune jointly. (ii) Prefer δ ≈ 2–5× the measured RMS vibrational amplitude at your `k×Δt` timescale.

### Citations & Data Anchors (for the ranges)

- Debye–Waller / ADP overview linking ⟨u²⟩ to thermal vibration amplitude (solids).
- Self‑diffusion in **liquid Al** from ab‑initio MD, showing D of order 10⁻⁹ m²/s near 1000 K; consistent with experiments.
- Typical self‑diffusion magnitudes in **liquid metals near melting** summarized around 10⁻⁹–3×10⁻⁹ m²/s.
- QENS methodology for measuring D in liquid metals (picosecond dynamics).
- Example tutorial for **liquid Cu** reporting D ≈ 0.89 Å²/ps (≈ 8.9×10⁻⁹ m²/s).

> Use these D values with √(6Dt) to sanity‑check a proposed δ for your chosen `k×Δt`. For instance, with D = 5×10⁻⁹ m²/s and t = 10 fs, √(6Dt) ≈ 0.055 Å → a δ around **0.05 Å** is physically sensible for molten metals.

---
