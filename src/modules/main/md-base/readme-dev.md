# Stitch Audit

Evaluates boundary continuity between adjacent MD trajectory fragments (`vasprun.xml`, `vasprun_1.xml`, ...).

## What it does

1. Reads first and last frame from each vasprun file
2. Checks velocity existence and temperature proxy (kinetic energy) per segment
3. Evaluates each boundary:
   - Cell consistency (norm + volume difference)
   - PBC-aware minimal-image atomic displacement
4. Outputs JSON with decisions and diagnostics

## Output

**`stitch_summary.json`** contains:

- **segments**: metadata (path, nsteps, natoms)
- **boundaries**: per-boundary metrics and decision labels
  - `decision`: `STITCH`, `NO_STITCH`, `DUPLICATE_FRAME`, or `UNCERTAIN`
  - `rms_disp`, `max_disp`, `p95_disp`: displacement statistics
  - `topk_atoms`: top-20 atoms with largest displacements (for diagnostics)
  - `cell_diff`, `vol_diff_frac`: cell consistency metrics
- **thresholds**: settings used for reproducibility

## Decision Labels

| Label             | Meaning                                                |
| ----------------- | ------------------------------------------------------ |
| `STITCH`          | Smooth continuation (rms < 0.05 Å)                     |
| `NO_STITCH`       | Discontinuity or cell mismatch                         |
| `DUPLICATE_FRAME` | Same positions, same velocities (drop one frame later) |
| `UNCERTAIN`       | RMS displacement between thresholds (inspect manually) |

## Diagnostics from Log Output

Console prints:

- Velocity sanity checks per segment (KE values, ratio warnings)
- Boundary evaluation summary (decision + displacement metrics)

Example:

```
14:32:15 [__main__] INFO - Found 6 segments
14:32:15 [__main__] INFO - [0] vasprun.xml: 172 steps, 510 atoms
14:32:16 [__main__] INFO - Segment 0: KE_first=0.1234, KE_last=0.1245
14:32:16 [__main__] INFO - KE ratio 0-1: 1.05
14:32:17 [__main__] INFO - [0 → 1] NO_STITCH           | rms=0.245630 Å | max=0.891233 Å
14:32:17 [__main__] INFO - [1 → 2] STITCH              | rms=0.008934 Å | max=0.045621 Å
```

## Tuning Thresholds

Edit `THRESHOLDS` dict:

```python
THRESHOLDS = {
    "cell_diff_tol": 0.05,      # Ångströms
    "vol_diff_tol": 0.01,       # fraction
    "rms_stitch": 0.05,         # Ångströms
    "rms_no_stitch": 0.1,       # Ångströms
}
```

## Minimal Design

- 232 lines, no CLI, no external abstractions
- Pure logging (no data manipulation in sanity checks)
- One pass: read → check → evaluate → output
