from dataclasses import dataclass

from monty.serialization import loadfn
from pymatgen.core import Structure

# from pymatgen.io.vasp import Incar
from pymatgen.io.vasp.sets import VaspInputSet


def manual_init_NaCl_structure():
    from pymatgen.core import Lattice, Structure

    # Create the NaCl structure
    lattice = Lattice.cubic(5.64)  # Lattice parameter for NaCl
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    structure.sort()  # this groups all elements of a single type together, makes the POTCAR easier

    print(structure)


def manual_potcar():
    # Create and write the POTCAR file
    from pymatgen.io.vasp import Potcar

    # Specify the POTCAR symbols for NaCl --> these should be in the same order as POSCAR
    symbols = ["Na_pv", "Cl"]

    potcar = Potcar(
        symbols, functional="PBE_64"
    )  # remember that your functional should match your installed POTCARs
    potcar.write_file("POTCAR")

    print("POTCAR")


SETS_DIR = "sets"


def _load_yaml_config(fname):
    config = loadfn(f"{SETS_DIR}/{fname}.yaml")
    if "PARENT" in config:
        parent_config = _load_yaml_config(config["PARENT"])
        for k, v in parent_config.items():  # pyright: ignore
            if k not in config:
                config[k] = v
            elif isinstance(v, dict):
                v_new = config.get(k, {})
                v_new.update(v)
                config[k] = v_new
    return config


@dataclass
class MPRelaxSet2025_3_10(VaspInputSet):
    """
    Implementation of VaspInputSet utilizing parameters in the public
    Materials Project. Typically, the pseudopotentials chosen contain more
    electrons than the MIT parameters, and the k-point grid is ~50% more dense.
    The LDAUU parameters are also different due to the different PSPs used,
    which result in different fitted values.

    Args:
        structure (Structure): The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set separately before
            the inputs are generated.
        **kwargs: Keywords supported by VaspInputSet.
    """

    CONFIG = _load_yaml_config("MPRelaxSet_v2025.3.10")


if __name__ == "__main__":
    name = "Zr4_Cu2_mp-193_"

    # Load your structure
    structure = Structure.from_file(f"input_{name}/POSCAR")

    # Create an MPRelaxSet instance
    relax_set = MPRelaxSet2025_3_10(structure)

    # Generate only POSCAR
    poscar = relax_set.potcar
    poscar.write_file(f"{name}.potcar")

    # Generate only INCAR
    incar = relax_set.incar
    incar.write_file(f"{name}.incar")

    # # Modify INCAR parameters manually
    # incar_dict = relax_set.incar.as_dict()
    # incar_dict.update({"ISPIN": 2, "NCORE": 4})  # Add or modify parameters
    # modified_incar = Incar.from_dict(incar_dict)
    # modified_incar.write_file(f"{name}.incar_mod")
