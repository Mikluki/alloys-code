import logging
from pathlib import Path

from ..energy.energy_source import EnergySource
from .base import BaseNN
from .decorators import calculator_for

LOGGER = logging.getLogger(__name__)
MODELS_DIR = "uvpy/models/"


@calculator_for(EnergySource.DPA31)
class Dpa31(BaseNN):
    """DeePMD (Deep Potential Molecular Dynamics) calculator."""

    def _initialize_custom_subclass(
        self,
        **kwargs,
    ) -> None:
        """Initialize DeePMD calculator.

        Args:
            **kwargs: Additional arguments passed to DP()
        """
        from deepmd.calculator import DP

        model_path = (
            self.checkpoint_path or Path.home() / MODELS_DIR / "dpa-3.1-3m-ft.pth"
        )

        LOGGER.info(f"{__class__.__name__} :: used model_path: `{model_path}`")

        self._ase_calculator = DP(
            model_path,
            **kwargs,
        )


@calculator_for(EnergySource.GRACE_2L_OAM_L)
class Grace2lOamL(BaseNN):
    """GRACE-2L-OAM-L neural network potential."""

    def _initialize_custom_subclass(
        self,
        **kwargs,
    ) -> None:
        """Initialize GRACE calculator.

        Args:
            **kwargs: Additional arguments passed to grace_fm()
        """
        from tensorpotential.calculator import grace_fm

        model_name = self.checkpoint_path or "GRACE-2L-OMAT-large-ft-AM"

        LOGGER.info(f"{__class__.__name__} :: used model_name: `{model_name}`")

        self._ase_calculator = grace_fm(
            model_name,
            **kwargs,
        )


@calculator_for(EnergySource.NEQUIP)
class Nequip(BaseNN):
    """NequIP neural network potential using GPU-compiled models."""

    def _initialize_custom_subclass(
        self,
        **kwargs,
    ) -> None:
        """
        Args:
            compile_path: Path to the compiled NequIP model (.pt2 file)
            **kwargs: Additional arguments passed to NequIPCalculator.from_compiled_model()

        Returns:
            NequIPCalculator: NequIP calculator instance
        """
        from nequip.ase import NequIPCalculator

        checkpoint = (
            self.checkpoint_path
            or Path.home() / MODELS_DIR / "mir-group__NequIP-OAM-L__0.1.nequip.pt2"
        )

        LOGGER.info(f"{__class__.__name__} :: used checkpoint_path: `{checkpoint}`")

        self._ase_calculator = NequIPCalculator.from_compiled_model(
            compile_path=checkpoint,
            device=self.device,
            **kwargs,
        )


@calculator_for(EnergySource.ALLEGRO)
class Allegro(BaseNN):
    """Allegro neural network potential using GPU-compiled models."""

    def _initialize_custom_subclass(
        self,
        **kwargs,
    ) -> None:
        """
        Args:
            compile_path: Path to the compiled Allegro model (.pt2 file)
            **kwargs: Additional arguments passed to NequIPCalculator.from_compiled_model()

        Returns:
            NequIPCalculator: Allegro calculator instance
        """
        from nequip.ase import NequIPCalculator

        checkpoint = (
            self.checkpoint_path
            or Path.home() / MODELS_DIR / "mir-group__Allegro-OAM-L__0.1.nequip.pt2"
        )

        LOGGER.info(f"{__class__.__name__} :: used checkpoint_path: `{checkpoint}`")

        self._ase_calculator = NequIPCalculator.from_compiled_model(
            compile_path=checkpoint,
            device=self.device,
            **kwargs,
        )


@calculator_for(EnergySource.ESEN_30M_OAM)
class Esen(BaseNN):
    """ESᴇN (Equivariant Spherical Networks) calculator.

    Warning: Requires fairchem-core==1.10 (v1 API).
    FairChem v2 broke compatibility with v1 checkpoints.
    """

    def _initialize_custom_subclass(self, **kwargs) -> None:
        """Initialize ESᴇN calculator."""
        from fairchem.core import OCPCalculator

        # Use provided checkpoint or default
        checkpoint = (
            self.checkpoint_path or Path.home() / MODELS_DIR / "esen_30m_oam.pt"
        )

        LOGGER.info(f"{__class__.__name__} :: used checkpoint_path: `{checkpoint}`")

        # Convert device string to cpu boolean
        cpu = self.device == "cpu"

        self._ase_calculator = OCPCalculator(
            checkpoint_path=checkpoint,
            cpu=cpu,
            **kwargs,
        )


@calculator_for(EnergySource.MACE)
class Mace(BaseNN):
    """MACE (Multi Atomic Cluster Expansion) calculator."""

    def _initialize_custom_subclass(
        self,
        dispersion: bool = False,
        default_dtype: str = "float64",
        **kwargs,
    ) -> None:
        """Initialize MACE calculator.

        Args:
            dispersion: Whether to include dispersion corrections
            default_dtype: Default data type for calculations
        """
        from mace.calculators import mace_mp

        # Use provided checkpoint or default
        model_path = (
            self.checkpoint_path
            or Path.home() / MODELS_DIR / "MACE-matpes-pbe-omat-ft.model"
        )

        LOGGER.info(f"{__class__.__name__} :: used model_path: `{model_path}`")

        self._ase_calculator = mace_mp(
            model=model_path,
            device=self.device,
            dispersion=dispersion,
            default_dtype=default_dtype,
            **kwargs,
        )


@calculator_for(EnergySource.MACE_MPA_0)
class Mace_mpa_0(BaseNN):
    """MACE (Multi Atomic Cluster Expansion) calculator."""

    def _initialize_custom_subclass(
        self,
        dispersion: bool = False,
        default_dtype: str = "float64",
        **kwargs,
    ) -> None:
        """Initialize MACE_MPA_0 calculator.

        Args:
            dispersion: Whether to include dispersion corrections
            default_dtype: Default data type for calculations
        """
        from mace.calculators import mace_mp

        # Use provided checkpoint or default
        model_path = (
            self.checkpoint_path or Path.home() / MODELS_DIR / "mace-mpa-0-medium.model"
        )

        self._ase_calculator = mace_mp(
            model=model_path,
            device=self.device,
            dispersion=dispersion,
            default_dtype=default_dtype,
            **kwargs,
        )


@calculator_for(EnergySource.ORBV3)
class Orbv3(BaseNN):
    """ORB v3 Conservative calculator with effectively unlimited neighbors."""

    def _initialize_custom_subclass(
        self,
        precision: str = "float64",
        **kwargs,
    ) -> None:
        """Initialize ORB v3 calculator.

        Args:
            precision: Model precision ("float32-high", "float64", etc.)
        """
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        # Load pretrained model
        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=self.device,
            precision=precision,
            **kwargs,
        )

        # Wrap in ORB calculator
        self._ase_calculator = ORBCalculator(orbff, device=self.device)


@calculator_for(EnergySource.SEVENNET)
class SevenNet(BaseNN):
    """SevenNet Calculator with support for multi-modal models."""

    def _initialize_custom_subclass(
        self,
        model: str = "7net-mf-ompa",
        modal: str | None = "mpa",
        file_type: str = "checkpoint",
        enable_cueq: bool = False,
        **kwargs,
    ) -> None:
        """Initialize SevenNet calculator.

        Args:
            model: Model name or path (7net-mf-ompa, 7net-omat, 7net-l3i5, 7net-0)
            modal: Modal for multi-modal models (e.g., 'mpa', 'omat24' for 7net-mf-ompa)
            file_type: File type ('checkpoint', 'torchscript', 'model_instance')
            enable_cueq: Enable cuEquivariant acceleration
        """
        from sevenn.calculator import SevenNetCalculator

        # Use checkpoint_path if provided, otherwise use model parameter
        model_or_path = self.checkpoint_path or model

        self._ase_calculator = SevenNetCalculator(
            model=model_or_path,
            device=self.device,
            modal=modal,
            file_type=file_type,
            enable_cueq=enable_cueq,
            **kwargs,
        )
