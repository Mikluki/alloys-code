from pymatgen.io.vasp.outputs import Outcar, Vasprun

# Path to your OUTCAR and vasprun.xml files
outcar_path = "files/staer25-04-14/Cu3_Ni3_mp-1225687_/OUTCAR"
vasprun_path = "files/staer25-04-14/Cu3_Ni3_mp-1225687_/vasprun.xml"

# Parse the OUTCAR file
outcar = Outcar(outcar_path)

# Run stats
print(f"Run statistics: {outcar.run_stats}\n")
print(f"{outcar.run_stats.get('Elapsed time (sec)')}")

# In outcar final energy is directly available
relaxed_energy_final = outcar.final_energy
print(f"Final relaxed energy: {relaxed_energy_final}")

### Vasprun
vasprun = Vasprun(vasprun_path)

relaxed_energy_init = vasprun.ionic_steps[0]["e_fr_energy"]
print(f"Initial relaxed energy: {relaxed_energy_init}")

# For cell volume information
cell_volume_init = vasprun.structures[0].volume
cell_volume_final = vasprun.structures[-1].volume
print(f"Initial cell volume: {cell_volume_init}")
print(f"Final cell volume: {cell_volume_final}")

# Atoms
print(f"\nvasprun.structures[-1] {vasprun.structures[-1]}")

from datetime import datetime
from pathlib import Path

mtime = Path(outcar_path).stat().st_mtime
print(datetime.fromtimestamp(mtime))
# if self.outcar_path.exists():
#     mtime = self.outcar_path.stat().st_mtime
#     return datetime.fromtimestamp(mtime)
#
# return None
