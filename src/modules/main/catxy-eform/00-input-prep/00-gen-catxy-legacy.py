import itertools
import logging
from pathlib import Path

from vsf.generate.generator_binary import GeneratorBinaryPrototype, Prototypes
from vsf.logging import setup_logging
from vsf.transform.poscar_organizer import PoscarOrganizer

LOGGER = setup_logging(log_file="x-generate.log", console_level=logging.INFO)

# Define the MP ID to element mapping
mp_to_element = {
    "mp-67": "Sc",
    "mp-72": "Ti",
    "mp-146": "V",
    "mp-90": "Cr",
    "mp-35": "Mn",
    "mp-13": "Fe",
    "mp-102": "Co",
    "mp-23": "Ni",
    "mp-30": "Cu",
    "mp-79": "Zn",
    "mp-112": "Y",
    "mp-131": "Zr",
    "mp-75": "Nb",
    "mp-129": "Mo",
    "mp-113": "Tc",
    "mp-33": "Ru",
    "mp-74": "Rh",
    "mp-2": "Pd",
    "mp-8566": "Ag",
    "mp-94": "Cd",
    "mp-145": "Lu",
    "mp-103": "Hf",
    "mp-569794": "Ta",
    "mp-91": "W",
    "mp-1186901": "Re",
    "mp-49": "Os",
    "mp-101": "Ir",
    "mp-126": "Pt",
    "mp-81": "Au",
    "mp-10861": "Hg",
}


# Generate bimetallic structures from template
def generate_bimetallic_structures(output_path, prototype, include_reverse):
    """Generate bimetallic structures using all combinations of elements."""
    print("Generating bimetallic structures...")

    # Get all unique element pairs (combinations)
    elements = list(mp_to_element.values())
    element_pairs = list(itertools.combinations(elements, 2))
    print(element_pairs)

    # Initialize structure assembler with template
    assembler = GeneratorBinaryPrototype.from_template(prototype)

    # Generate structures for each pair
    all_structures = []
    for elem1, elem2 in element_pairs:
        structures = assembler.substitute_atoms(
            elem1, elem2, include_reverse=include_reverse
        )
        all_structures.extend(structures)

    # Save all generated structures
    assembler.save_structures(all_structures, output_path, prefix=prototype.label)
    print(f"Generated {len(all_structures)} bimetallic structures in {output_path}")

    return all_structures


if __name__ == "__main__":
    base_dir = Path("catxy-init")
    num_structures = 2

    prototypes = [
        # Prototypes.B1,
        Prototypes.B2,
        # Prototypes.B3,
        # Prototypes.B4,
        # Prototypes.B11,
        # Prototypes.B19,
        # Prototypes.L10,
        # Prototypes.L11,
    ]

    for p in prototypes:
        print(f"=== {p.label} ===")
        output_path = Path(f"{p.label}-entries", "poscars")
        output_path.mkdir(parents=True, exist_ok=True)

        generate_bimetallic_structures(output_path, p, include_reverse)

        # Initialize entries
        entries = initialize_entries(output_path, prototype.name)

    # Organize input paths to dedicated dirs
    key = "POSCAR"
    poscar_paths = [p for p in base_dir.rglob("*") if key in p.name]

    starting_id = 2000000
    organizer = PoscarOrganizer.from_starting_id(starting_id)
    organized_dirs = organizer.organize_poscar_list(poscar_paths, base_dir, "rand")
