import json
import os
from itertools import combinations

from mp_api.client import MPRester

# Load your API key
MP_API_KEY = str(os.getenv("MP_API_KEY"))

# Define target elements and their corresponding Materials Project IDs
struct_pure_Byung_dict = {
    "Sc": "mp-67",  # hcp #hp2
    "Ti": "mp-72",
    "V": "mp-146",
    "Cr": "mp-90",
    "Mn": "mp-35",
    "Fe": "mp-13",
    "Co": "mp-102",
    "Ni": "mp-23",
    "Cu": "mp-30",
    "Zn": "mp-79",
    "Y": "mp-112",  # hcp
    "Zr": "mp-131",  # hcp
    "Nb": "mp-75",
    "Mo": "mp-129",
    "Tc": "mp-113",  # hcp
    "Ru": "mp-33",
    "Rh": "mp-74",
    "Pd": "mp-2",
    "Ag": "mp-124",
    "Cd": "mp-94",  # hcp
    "Lu": "mp-145",  # hcp
    "Hf": "mp-103",
    "Ta": "mp-50",  # cl
    "W": "mp-91",  # cl
    "Re": "mp-8",  # hcp
    "Os": "mp-49",  # hcp
    "Ir": "mp-101",
    "Pt": "mp-126",
    "Au": "mp-81",
    "Hg": "mp-10861",  # hR1
}

# Generate all possible 3-element combinations of target elements
element_combinations = combinations(struct_pure_Byung_dict.keys(), 2)

# Initialize the output dictionary
target_comp = {}

# Query Materials Project for each combination
with MPRester(api_key=MP_API_KEY) as mpr:
    for comb in element_combinations:
        # Create a chemsys string in the format "X-Y-Z"
        chemsys = "-".join(comb)
        print(f"\nXXX Chemsys {chemsys}")

        # Search for thermo documents for the given chemical system
        thermo_docs = mpr.materials.thermo.search(chemsys=[chemsys])

        # Extract data and populate the target_comp dictionary
        for doc in thermo_docs:
            formula = doc.formula_pretty  # pyright: ignore
            material_id = doc.material_id  # pyright: ignore
            target_comp[formula] = material_id


# Save the results to JSON files
output_file = "chemsys-mpids.json"
with open(output_file, "w") as json_file:
    json.dump(target_comp, json_file, indent=4)

metadata_file = "search_metadata.json"
metadata = {
    "struct_pure_dict": struct_pure_Byung_dict,
    "total_target_comp": len(target_comp),
}
with open(metadata_file, "w") as json_file:
    json.dump(metadata, json_file, indent=4)

# Print or save the result
print(f"Data saved to {output_file} and metadata saved to {metadata_file}")
