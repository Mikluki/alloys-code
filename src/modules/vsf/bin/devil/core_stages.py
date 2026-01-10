"""
core_stages.py
Contains Stage Implementations
"""

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from core import DirectoryResolver, Stage, StageContext, StageResult, Workflow

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Concrete Stage Implementations
# ============================================================================


@dataclass
class SubmitStageConfig:
    """Configuration for VASP submission"""

    target_dir_expr: str
    ntasks: int
    ncore: int
    kpar: int
    algo: str
    partition: Optional[str] = None
    cpus_per_task: Optional[int] = None
    nodes: int = 1
    nodelist: str = ""


class SubmitStage(Stage):
    """Submits a VASP job using vsf-submit-job.py"""

    def __init__(self, name: str, config: SubmitStageConfig):
        super().__init__(name)
        self.config = config

    def execute(self, context: StageContext) -> StageResult:
        # Resolve target directory
        target_dir = DirectoryResolver.resolve(self.config.target_dir_expr, context)

        # Validate VASP directory
        if not self._validate_vasp_directory(target_dir):
            return StageResult(
                success=False,
                target_dir=target_dir,
                error_message=f"Invalid VASP directory (missing input files): {target_dir}",
            )

        # Build submission command
        cmd = self._build_submit_command(target_dir, context)

        # Execute submission
        try:
            LOGGER.info(f"Submitting job for {target_dir}")
            LOGGER.debug(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=30
            )

            # Parse job ID from output
            job_id = self._parse_job_id(result.stdout)

            if not job_id:
                LOGGER.warning(f"Could not parse job ID from output:\n{result.stdout}")
                return StageResult(
                    success=False,
                    target_dir=target_dir,
                    error_message="Could not parse job ID from submission output",
                )

            LOGGER.info(f"✓ Submitted job {job_id} for {target_dir}")

            return StageResult(
                success=True,
                target_dir=target_dir,
                job_id=job_id,
                metadata={"submit_output": result.stdout.strip()},
            )

        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Submission failed: {e.stderr}")
            return StageResult(
                success=False,
                target_dir=target_dir,
                error_message=f"Submission failed: {e.stderr}",
            )
        except subprocess.TimeoutExpired:
            return StageResult(
                success=False,
                target_dir=target_dir,
                error_message="Submission timed out",
            )
        except Exception as e:
            LOGGER.exception(f"Unexpected error during submission")
            return StageResult(
                success=False,
                target_dir=target_dir,
                error_message=f"Unexpected error: {e}",
            )

    def requires_job_completion(self) -> bool:
        return True

    def _validate_vasp_directory(self, path: Path) -> bool:
        """Check if directory has required VASP input files"""
        required = ["INCAR", "POSCAR", "POTCAR", "KPOINTS"]
        missing = [f for f in required if not (path / f).exists()]
        if missing:
            LOGGER.error(f"Missing required files in {path}: {missing}")
            return False
        return True

    def _build_submit_command(
        self, target_dir: Path, context: StageContext
    ) -> List[str]:
        """Build command for vsf-submit-job.py"""
        vasp_setup = context.global_config["vasp_setup"]

        cmd = [
            "vsf-submit-job.py",
            str(target_dir),
            str(self.config.ntasks),
            str(self.config.ncore),
            str(self.config.kpar),
            self.config.algo,
            "--vasp-setup",
            vasp_setup,
        ]

        if self.config.cpus_per_task:
            cmd.extend(["--cpus_per_task", str(self.config.cpus_per_task)])
        if self.config.partition:
            cmd.extend(["--partition", self.config.partition])
        if self.config.nodes != 1:
            cmd.extend(["--nodes", str(self.config.nodes)])
        if self.config.nodelist:
            cmd.extend(["--nodelist", self.config.nodelist])

        return cmd

    def _parse_job_id(self, output: str) -> Optional[str]:
        """Parse job ID from vsf-submit-job.py output"""
        match = re.search(r"with ID (\d+)", output)
        return match.group(1) if match else None


@dataclass
class CopyStageConfig:
    """Configuration for file copying"""

    source_files: List[str]
    target_dir_expr: str
    rename_map: Dict[str, str] = field(default_factory=dict)
    create_target: bool = True


class CopyStage(Stage):
    """Copies files between directories"""

    VASP_CONTINUATION_FILES = [
        "CONTCAR",
        "WAVECAR",
        "INCAR",
        "KPOINTS",
        "POTCAR",
    ]

    def __init__(self, name: str, config: CopyStageConfig):
        super().__init__(name)
        self.config = config

    def execute(self, context: StageContext) -> StageResult:
        if not context.previous_target_dir:
            return StageResult(
                success=False,
                target_dir=Path("."),
                error_message="CopyStage requires previous stage",
            )

        source_dir = context.previous_target_dir
        target_dir = DirectoryResolver.resolve(self.config.target_dir_expr, context)

        # Create target directory
        if self.config.create_target:
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Created directory: {target_dir}")
            except Exception as e:
                return StageResult(
                    success=False,
                    target_dir=target_dir,
                    error_message=f"Failed to create target dir: {e}",
                )

        # Copy files
        copied_files = []
        missing_files = []
        failed_files = []

        for src_file in self.config.source_files:
            src_path = source_dir / src_file

            if not src_path.exists():
                missing_files.append(src_file)
                LOGGER.warning(f"File {src_file} not found in {source_dir}, skipping")
                continue

            dst_name = self.config.rename_map.get(src_file, src_file)
            dst_path = target_dir / dst_name

            try:
                shutil.copy2(src_path, dst_path)
                copied_files.append(f"{src_file} → {dst_name}")
                LOGGER.debug(f"Copied {src_file} → {dst_path}")
            except Exception as e:
                failed_files.append(f"{src_file}: {e}")
                LOGGER.error(f"Failed to copy {src_file}: {e}")

        # Success if we copied at least one file
        success = len(copied_files) > 0

        if success:
            LOGGER.info(
                f"✓ Copied {len(copied_files)} files: {source_dir} → {target_dir}"
            )

        if missing_files:
            LOGGER.info(f"Missing files (skipped): {missing_files}")

        return StageResult(
            success=success,
            target_dir=target_dir,
            metadata={
                "copied": copied_files,
                "missing": missing_files,
                "failed": failed_files,
            },
            error_message=None if success else "No files were copied",
        )

    def requires_job_completion(self) -> bool:
        return False

    @classmethod
    def for_vasp_continuation(cls, name: str, target_dir_expr: str) -> "CopyStage":
        """Factory: Create stage for VASP relaxation continuation"""
        config = CopyStageConfig(
            source_files=cls.VASP_CONTINUATION_FILES,
            target_dir_expr=target_dir_expr,
            rename_map={"CONTCAR": "POSCAR"},
        )
        return cls(name, config)


class ValidationStage(Stage):
    """Validates VASP calculation completed successfully"""

    def __init__(self, name: str = "validate"):
        super().__init__(name)

    def execute(self, context: StageContext) -> StageResult:
        if not context.previous_target_dir:
            return StageResult(
                success=False,
                target_dir=Path("."),
                error_message="ValidationStage requires previous stage",
            )

        target_dir = context.previous_target_dir
        outcar = target_dir / "OUTCAR"

        if not outcar.exists():
            return StageResult(
                success=False, target_dir=target_dir, error_message="OUTCAR not found"
            )

        # Check for completion marker
        try:
            with open(outcar) as f:
                content = f.read()

            if "Elapsed time (sec):" not in content:
                return StageResult(
                    success=False,
                    target_dir=target_dir,
                    error_message="VASP calculation did not complete (no elapsed time marker)",
                )

            # Extract elapsed time
            match = re.search(r"Elapsed time \(sec\):\s+([\d.]+)", content)
            elapsed_time = float(match.group(1)) if match else None

            if elapsed_time:
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                seconds = int(elapsed_time % 60)
                LOGGER.info(
                    f"✓ VASP completed in {hours:02d}:{minutes:02d}:{seconds:02d}"
                )

            return StageResult(
                success=True,
                target_dir=target_dir,
                metadata={"elapsed_time_sec": elapsed_time},
            )

        except Exception as e:
            return StageResult(
                success=False,
                target_dir=target_dir,
                error_message=f"Failed to read OUTCAR: {e}",
            )

    def requires_job_completion(self) -> bool:
        return False


# ============================================================================
# Workflow Implementations
# ============================================================================


def create_simple_workflow(
    ntasks: int, ncore: int, kpar: int, algo: str, **slurm_kwargs
) -> Workflow:
    """Single relaxation workflow"""

    submit_config = SubmitStageConfig(
        target_dir_expr="{source_dir}",
        ntasks=ntasks,
        ncore=ncore,
        kpar=kpar,
        algo=algo,
        **slurm_kwargs,
    )

    return Workflow(
        name="simple",
        description="Single VASP relaxation",
        stages=[SubmitStage("relaxation", submit_config), ValidationStage("validate")],
    )


def create_double_relaxation_workflow(
    ntasks: int, ncore: int, kpar: int, algo: str, **slurm_kwargs
) -> Workflow:
    """Double relaxation workflow with validation"""

    submit_config_1 = SubmitStageConfig(
        target_dir_expr="{source_dir}",
        ntasks=ntasks,
        ncore=ncore,
        kpar=kpar,
        algo=algo,
        **slurm_kwargs,
    )

    submit_config_2 = SubmitStageConfig(
        target_dir_expr="{source_dir}_relax2",
        ntasks=ntasks,
        ncore=ncore,
        kpar=kpar,
        algo=algo,
        **slurm_kwargs,
    )

    return Workflow(
        name="double_relaxation",
        description="Two-stage relaxation: relax → validate → copy → relax → validate",
        stages=[
            SubmitStage("relax1", submit_config_1),
            ValidationStage("validate1"),
            CopyStage.for_vasp_continuation(
                "copy_for_relax2", target_dir_expr="{source_dir}_relax2"
            ),
            SubmitStage("relax2", submit_config_2),
            ValidationStage("validate2"),
        ],
    )
