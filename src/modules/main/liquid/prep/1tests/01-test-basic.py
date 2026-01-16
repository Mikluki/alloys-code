import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alchemlyb.preprocessing.subsampling import (
    equilibrium_detection,  # removes burn-in + subsamples
)
from alchemlyb.preprocessing.subsampling import (
    statistical_inefficiency,  # subsamples only
)
from ase.geometry import find_mic
from ase.io.vasp import read_vasp_xdatcar

# --- 1) Load VASP MD
atoms_list = read_vasp_xdatcar("XDATCAR", index=":")  # pyright: ignore
T = len(atoms_list)
assert T >= 3, "Need at least 3 frames"

# --- 2) Build a scalar series: per-step MSD with minimum-image (PBC-safe)
cell = atoms_list[0].get_cell()
pbc = atoms_list[0].get_pbc()

msd_step = []
for t in range(1, T):
    R2 = atoms_list[t].get_positions()
    R1 = atoms_list[t - 1].get_positions()
    d = R2 - R1

    # Use per-frame cell/pbc in case they change (NPT, barostat, etc.)
    cell = atoms_list[t].get_cell()
    pbc = atoms_list[t].get_pbc()

    # Version-agnostic unpacking
    res = find_mic(d, cell, pbc)
    d_mic = (
        res[0] if isinstance(res, tuple) else res
    )  # some ASE versions return (d_mic, ...), others just d_mic

    msd_step.append((d_mic**2).sum(axis=1).mean())

msd_step = np.asarray(msd_step)  # length = T-1

# --- 3) Wrap into pandas with a 'time' index (required by alchemlyb helpers)
time = pd.Index(np.arange(len(msd_step), dtype=float), name="time")
series = pd.Series(msd_step, index=time, name="msd_step")

# df can be anything you want to slice alongside (here: the original next-frame indices)
df = pd.DataFrame(
    {"frame": np.arange(1, T)}, index=time
)  # each msd_step[t] -> frame t+1

# --- Option A: remove burn-in + subsample (recommended)
df_eq_sub = equilibrium_detection(df, series=series, drop_duplicates=True, sort=True)

# --- Option B: only subsample (no burn-in removal)
# df_eq_sub = statistical_inefficiency(df, series=series, drop_duplicates=True, sort=True)

# --- 4) Get the *trajectory frame indices* to keep (0-based)
kept_frames = df_eq_sub["frame"].to_numpy()  # these are already (t+1)
kept_frames_0based = kept_frames  # 0..T-1, since 'frame' was 1..T-1

print(f"Keeping {len(kept_frames_0based)} frames out of {T}.")
print("Frame indices to keep (0-based):", kept_frames_0based.tolist())

# Example: gather decorrelated atoms objects
decorrelated_atoms = [atoms_list[i] for i in kept_frames_0based]


plt.plot(msd_step, label="Per-step MSD")
plt.axvline(36, color="red", linestyle="--", label="Detected burn-in")
plt.legend()
plt.xlabel("Frame")
plt.ylabel("Scalar (MSD)")
plt.show()
