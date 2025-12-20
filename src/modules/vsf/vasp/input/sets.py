from dataclasses import dataclass
from importlib.resources import files

from monty.serialization import loadfn
from pymatgen.io.vasp.sets import VaspInputSet

# Define the path to the PMG sets directory
MPG_SETS_DIR = files("vsf.vasp.input").joinpath("PMGsets")


def load_yaml_config(fname: str) -> dict:
    """
    Load a YAML configuration file with inheritance support.

    Args:
        fname: Name of the YAML file without extension

    Returns:
        Merged configuration dictionary
    """
    config = loadfn(f"{MPG_SETS_DIR}/{fname}.yaml")
    if "PARENT" in config:
        parent_config = load_yaml_config(config["PARENT"])
        for k, v in parent_config.items():
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
        structure: The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set
            separately before the inputs are generated.
        **kwargs: Keywords supported by VaspInputSet.
    """

    CONFIG = load_yaml_config("MPRelaxSet_v2025.3.10")
