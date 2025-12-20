from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

# Import as a type reference only
if TYPE_CHECKING:
    from ..calculators.base import BaseNN


class EnergySource(Enum):
    MP = "materials_project"

    ESEN_30M_OAM = "ESEN_30M_OAM"
    MACE = "MACE"  # pbe-omat
    MACE_MPA_0 = "MACE_MPA_0"  # DFT (PBE+U)
    ORBV3 = "ORBV3"
    SEVENNET = "SEVENNET"

    NEQUIP = "NEQUIP"
    ALLEGRO = "ALLEGRO"
    GRACE_2L_OAM_L = "GRACE_2L_OAM_L"
    DPA31 = "DPA31"

    VASP = "VASP"
    MD_LIQUID = "MD_LIQUID"

    # Instance methods (use self) - for convenience on specific instances
    def venv(self) -> str | None:
        """Get the virtual environment path for this energy source."""
        return META[self].venv

    def kind(self) -> str:
        """Get the kind (e.g., 'nn', 'dft') for this energy source."""
        return META[self].kind

    def is_neural_network(self) -> bool:
        """Check if this energy source is a neural network."""
        return META[self].kind == "nn"

    def is_dft(self) -> bool:
        """Check if this energy source is DFT-based."""
        return META[self].kind == "dft"

    def has_venv(self) -> bool:
        """Check if this energy source requires a virtual environment."""
        return META[self].venv is not None

    def note(self) -> str | None:
        """Get any additional notes for this energy source."""
        return META[self].note

    # Class methods - for discovery and filtering across all sources
    @classmethod
    def neural_networks(cls) -> list[EnergySource]:
        """Get all neural network energy sources."""
        return [s for s in cls if META[s].kind == "nn"]

    @classmethod
    def vasp_based(cls) -> list[EnergySource]:
        """Get all VASP/DFT-based energy sources."""
        return [s for s in cls if META[s].kind == "dft"]

    @classmethod
    def services(cls) -> list[EnergySource]:
        """Get all service-based energy sources."""
        return [s for s in cls if META[s].kind == "service"]

    @classmethod
    def without_venv(cls) -> list[EnergySource]:
        """Get all energy sources that don't require a virtual environment."""
        return [s for s in cls if META[s].venv is None]

    @classmethod
    def from_dict(cls, value: str) -> EnergySource:
        """Convert string to EnergySource enum. Raises ValueError if invalid."""
        return cls(value)

    def to_dict(self) -> str:
        """Convert enum to string for serialization."""
        return self.value

    def get_calculator(self, device: str = "cpu", auto_init: bool = True) -> "BaseNN":
        """Get the calculator instance for this energy source."""
        from ..calculators import get_calculator

        return get_calculator(self, device=device, auto_init=auto_init)


# Metadata model with validation
@dataclass(frozen=True)
class SourceMeta:
    venv: str | None  # None allowed for VASP*; otherwise non-empty string
    kind: str  # e.g. "nn", "dft"
    note: str | None = None

    def __post_init__(self) -> None:
        if self.venv is not None and not self.venv.strip():
            raise ValueError("venv must be None or a non-empty string")


def _load_venv_config() -> dict[str, str | None]:
    """
    Load venv overrides from ~/.energy_sources.yaml.

    Returns empty dict if file doesn't exist (silent).
    Raises error if file exists but is invalid YAML.
    """
    config_path = Path.home() / ".energy_sources.yaml"

    if not config_path.exists():
        return {}

    if yaml is None:
        warnings.warn(
            f"Config file {config_path} found but PyYAML is not installed. "
            "Install it with: pip install pyyaml"
        )
        return {}

    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse config file {config_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to read config file {config_path}: {e}") from e

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise RuntimeError(f"Config file {config_path} must contain a YAML dictionary")

    venvs = config.get("venvs", {})
    if not isinstance(venvs, dict):
        raise RuntimeError(f"Config file {config_path}: 'venvs' must be a dictionary")

    return venvs


def _expand_venv_path(path: str | None) -> str | None:
    """Expand ~ in venv path to absolute path."""
    if path is None:
        return None
    return str(Path(path).expanduser())


def _build_meta() -> dict[EnergySource, SourceMeta]:
    """Build META registry with defaults and config overrides."""
    # Default venv paths (using ~/)
    defaults = {
        EnergySource.ESEN_30M_OAM: ("~/.venvs/uv312eSEN", "nn"),
        EnergySource.MACE: ("~/.venvs/uv312mace", "nn"),
        EnergySource.MACE_MPA_0: ("~/.venvs/uv312mace", "nn"),
        EnergySource.SEVENNET: ("~/.venvs/uv312SevenNet", "nn"),
        EnergySource.ORBV3: ("~/.venvs/uv312ORBv3", "nn"),
        ### CUDA ONLY
        EnergySource.NEQUIP: ("~/.venvs/uv312nequip", "nn"),
        EnergySource.ALLEGRO: ("~/.venvs/uv312allegro", "nn"),
        ### added 25-10-30
        EnergySource.GRACE_2L_OAM_L: ("~/.venvs/uv312grace", "nn"),
        EnergySource.DPA31: ("~/.venvs/uv312dpa31", "nn"),
        ### DFT
        EnergySource.VASP: (None, "dft"),
        EnergySource.MD_LIQUID: (None, "dft"),
        ### materials_project
        EnergySource.MP: ("~/.venvs/py312", "service"),
    }

    # Load config overrides
    config_venvs = _load_venv_config()

    # Warn about unknown sources in config
    known_names = {source.value for source in EnergySource}
    for name in config_venvs:
        if name not in known_names:
            warnings.warn(
                f"Unknown energy source '{name}' in config file. "
                f"Known sources: {sorted(known_names)}"
            )

    # Build final META
    meta = {}
    for source, (default_venv, kind) in defaults.items():
        # Check if config overrides this source's venv
        config_venv = config_venvs.get(source.value, default_venv)
        # Expand ~ in path
        expanded_venv = _expand_venv_path(config_venv)
        meta[source] = SourceMeta(expanded_venv, kind)

    return meta


# Build and freeze META
_META_MUT = _build_meta()
META: Mapping[EnergySource, SourceMeta] = MappingProxyType(_META_MUT)


# Import-time consistency check
_missing = [m for m in EnergySource if m not in META]
_extras = [k for k in META.keys() if k not in list(EnergySource)]
if _missing or _extras:
    raise RuntimeError(
        f"EnergySource META mismatch. Missing: {_missing}; Extras: {_extras}"
    )
