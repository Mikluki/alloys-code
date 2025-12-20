import pandas as pd
from mp_api.client import MPRester

with open("/home/mik/data/mp_api_key", "r") as file:
    api_key = file.read().strip()

with MPRester(api_key=api_key) as mpr:
    thermo_types = [
        "GGA_GGA+U",
        "GGA_GGA+U_R2SCAN",
        "R2SCAN",
    ]

    # chemsys = ["Ag"]
    dead_mpids = [
        "mp-1524302",
        "mp-1537768",
        "mp-1545421",
        "mp-1975010",
        "mp-2017103",
        "mp-2018976",
        "mp-2020519",
        "mp-2026160",
        "mp-2027032",
        "mp-2034504",
        "mp-2043976",
        "mp-2766475",
    ]
    thermo_docs = mpr.materials.thermo.search(
        material_ids=dead_mpids, thermo_types=thermo_types  # pyright: ignore
    )

thermo_fields = {
    "material_id",
    "thermo_id",
    # "formula_pretty",
    # "nsites",
    "energy_type",
    "energy_per_atom",
    # "uncorrected_energy_per_atom",
    "formation_energy_per_atom",
    "is_stable",
}


thermo_data = {
    getattr(entry, "formula_pretty"): {
        field: getattr(entry, field) for field in thermo_fields
    }
    for entry in thermo_docs
}
df = pd.DataFrame.from_dict(thermo_data, orient="index")
print(df)
