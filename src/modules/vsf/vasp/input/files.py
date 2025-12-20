import logging
from pathlib import Path
from typing import Type

from pymatgen.io.vasp import Poscar
from pymatgen.io.vasp.sets import VaspInputSet

from .sets import MPRelaxSet2025_3_10
from .writers import save_incar_from_set, save_kpoints_mueller, save_potcar_from_set

LOGGER = logging.getLogger(__name__)


class VaspInputFiles:
    """Manages VASP input files from a POSCAR file"""

    def __init__(self, poscar_path: Path, output_dir: Path | None = None):
        self.poscar_path = Path(poscar_path)
        self.output_dir = output_dir or self.poscar_path.parent
        self.name = f"{self.poscar_path.parent.name}/{self.poscar_path.name}"
        self._structure = None

    @property
    def structure(self):
        """Get structure from POSCAR file"""
        if self._structure is None:
            poscar = Poscar.from_file(self.poscar_path)
            self._structure = poscar.structure
        return self._structure

    def save_incar(
        self,
        input_set_source: Type[VaspInputSet] | str = MPRelaxSet2025_3_10,
        rewrite: bool = True,
        custom_incar_params: dict | None = {"NCORE": 4},
        use_blank_incar: bool = True,
        **kwargs,
    ) -> None:
        """
        Generate and write INCAR file using the specified input set source.

        Args:
            input_set_source: VaspInputSet class or YAML config name (default: MPRelaxSet2025_3_10)
            rewrite: Whether to overwrite existing file (default: True)
            custom_incar_params: Dictionary of custom INCAR parameters to override defaults
            use_blank_incar: If True, starts with blank INCAR using only MAGMOM from input set.
                           If False, uses full parameter set from input_set_source (default: True)
        """
        incar_path = self.output_dir / "INCAR"

        if not rewrite and incar_path.exists():
            return None

        save_incar_from_set(
            structure=self.structure,
            incar_name=incar_path.stem,
            incar_dir=self.output_dir,
            potcar_name="POTCAR",
            encut_factor=1.3,
            custom_incar_params=custom_incar_params,
            input_set_source=input_set_source,
            use_blank_incar=use_blank_incar,
            **kwargs,
        )

        LOGGER.info(f"For `{self.name}` INCAR saved: {self.output_dir.name}/INCAR")

    def save_potcar(
        self,
        input_set_source: Type[VaspInputSet] | str = MPRelaxSet2025_3_10,
        rewrite: bool = True,
    ) -> None:
        """
        Generate and write POTCAR file using the specified input set source.

        Args:
            input_set_source: VaspInputSet class or YAML config name (default: MPRelaxSet2025_3_10)
            rewrite: Whether to overwrite existing file (default: True)
        """
        potcar_path = self.output_dir / "POTCAR"

        if not rewrite and potcar_path.exists():
            return None

        save_potcar_from_set(
            structure=self.structure,
            potcar_name=potcar_path.stem,
            potcar_dir=self.output_dir,
            input_set_source=input_set_source,
        )

        LOGGER.info(f"For `{self.name}` POTCAR saved: {self.output_dir.name}/POTCAR")

    def save_kpoints(
        self,
        min_distance: float = 35.0,
        include_gamma: str = "auto",
        rewrite: bool = True,
    ) -> bool:
        """
        Generate and write KPOINTS file using Mueller method.

        Args:
            min_distance: Minimum distance parameter for k-point generation (default: 35.0)
            include_gamma: Whether to include gamma point, options: "TRUE", "FALSE", "AUTO"
            rewrite: Whether to overwrite existing file (default: True)
        """
        kpoints_path = self.output_dir / "KPOINTS"

        if not rewrite and kpoints_path.exists():
            LOGGER.info(
                f"For `{self.name}` [{self.output_dir.name}/KPOINTS] already exist"
            )
            return False

        success = save_kpoints_mueller(
            kpoints_path=kpoints_path,
            min_distance=min_distance,
            include_gamma=include_gamma,
            precalc_params={"HEADER": "VERBOSE", "WRITE_LATTICE_VECTORS": "True"},
        )

        LOGGER.info(f"For `{self.name}` KPOINTS saved: {self.output_dir.name}/KPOINTS")

        return success

    def save_all(
        self,
        input_set_source: Type[VaspInputSet] | str = MPRelaxSet2025_3_10,
        rewrite: bool = True,
        custom_incar_params: dict | None = {"NCORE": 4},
        use_blank_incar: bool = True,
        min_distance: float = 35.0,
        include_gamma: str = "auto",
        **kwargs,
    ) -> None:
        """
        Generate all VASP input files (INCAR, POTCAR, KPOINTS).

        Args:
            input_set_source: VaspInputSet class or YAML config name (default: MPRelaxSet2025_3_10)
            rewrite: Whether to overwrite existing files (default: True)
            custom_incar_params: Custom INCAR parameters (default: {"NCORE": 4})
            use_blank_incar: If True, starts with blank INCAR using only MAGMOM (default: True)
            min_distance: K-point generation parameter (default: 35.0)
            include_gamma: Gamma point inclusion (default: "auto")
        """
        self.save_potcar(input_set_source=input_set_source, rewrite=rewrite)

        # Incar gets ENCUT from potcars, thus potcars should be written first
        self.save_incar(
            input_set_source=input_set_source,
            rewrite=rewrite,
            custom_incar_params=custom_incar_params,
            use_blank_incar=use_blank_incar,
            **kwargs,
        )
        self.save_kpoints(
            min_distance=min_distance,
            include_gamma=include_gamma,
            rewrite=rewrite,
        )
