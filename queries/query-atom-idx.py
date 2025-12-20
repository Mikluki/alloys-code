import os

from mp_api.client import MPRester

# Load your API key
MP_API_KEY = str(os.getenv("MP_API_KEY"))

from mp_api.client import MPRester

chemsys = "Cr-Ni-Al"
thermo_types = [
    "GGA_GGA+U",
    # "GGA_GGA+U_R2SCAN",
    # "R2SCAN",
]
with MPRester(MP_API_KEY) as mpr:
    summaries = mpr.materials.thermo.search(
        chemsys=[chemsys], thermo_types=thermo_types
    )


threshold = 5  # Define your threshold here

for summary in summaries:
    composition = summary.composition
    print(f"Material ID: {summary.material_id}, Composition: {composition}")
    if all(amount < threshold for amount in composition.values()):
        print(f"Material ID: {summary.material_id}, Composition: {composition}")
