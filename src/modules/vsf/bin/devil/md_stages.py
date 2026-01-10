"""
md_stages.py
MD-specific workflow stages for iterative VASP runs with graceful shutdown.
Handles job submission, validation, backup versioning, and run preparation.
"""

import logging
import re
from pathlib import Path

from core import LoopingWorkflow, Stage, StageContext, StageResult

LOGGER = logging.getLogger(__name__)


# ============================================================================
# MD Submit Stage
# ============================================================================


class MDSubmitStage(Stage):
    """Submits VASP MD job"""

    def __init__(self, name: str = "md_submit"):
        super().__init__(name)

    def execute(self, context: StageContext) -> StageResult:
        """Submit MD job"""
        calc_dir = context.source_dir

        # Validate input files exist
        if not self._validate_vasp_directory(calc_dir):
            return StageResult(
                success=False,
                target_dir=calc_dir,
                error_message=f"Invalid VASP directory (missing input files): {calc_dir}",
            )

        # Import here to avoid circular dependency
        from .engine import SlurmInterface

        slurm = SlurmInterface()

        # Extract config
        if "md_config" not in context.global_config:
            return StageResult(
                success=False,
                target_dir=calc_dir,
                error_message="MD config not found in global_config",
            )

        md_config = context.global_config["md_config"]

        # Submit job
        try:
            job_id = slurm.submit_job(calc_dir, md_config)

            if not job_id:
                return StageResult(
                    success=False,
                    target_dir=calc_dir,
                    error_message="Could not parse job ID from submission output",
                )

            LOGGER.info(f"✓ Submitted MD job {job_id} for {calc_dir}")

            return StageResult(
                success=True,
                target_dir=calc_dir,
                job_id=job_id,
                metadata={"run_number": context.global_config.get("current_run", 0)},
            )

        except Exception as e:
            LOGGER.error(f"Job submission failed: {e}")
            return StageResult(
                success=False,
                target_dir=calc_dir,
                error_message=f"Submission failed: {e}",
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


# ============================================================================
# MD Validation Stage
# ============================================================================


class MDValidationStage(Stage):
    """Validates VASP MD job completion"""

    def __init__(self, name: str = "md_validate"):
        super().__init__(name)

    def execute(self, context: StageContext) -> StageResult:
        """Check if MD job completed"""
        calc_dir = context.source_dir

        # Check OUTCAR exists
        outcar = calc_dir / "OUTCAR"
        if not outcar.exists():
            return StageResult(
                success=False,
                target_dir=calc_dir,
                error_message="OUTCAR not found",
            )

        # Check for completion marker
        try:
            with open(outcar) as f:
                content = f.read()

            if "Elapsed time (sec):" not in content:
                return StageResult(
                    success=False,
                    target_dir=calc_dir,
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
                    f"✓ VASP MD completed in {hours:02d}:{minutes:02d}:{seconds:02d}"
                )

            return StageResult(
                success=True,
                target_dir=calc_dir,
                metadata={"elapsed_time_sec": elapsed_time},
            )

        except Exception as e:
            return StageResult(
                success=False,
                target_dir=calc_dir,
                error_message=f"Failed to read OUTCAR: {e}",
            )

    def requires_job_completion(self) -> bool:
        return False


# ============================================================================
# MD Backup Stage
# ============================================================================


class MDBackupStage(Stage):
    """Backs up versioned MD output files"""

    OUTPUT_FILES = ["OUTCAR", "CONTCAR", "OSZICAR", "vasprun.xml", "XDATCAR"]
    INPUT_FILES = ["INCAR", "POSCAR"]

    def __init__(self, name: str = "md_backup"):
        super().__init__(name)

    def execute(self, context: StageContext) -> StageResult:
        """Backup completed run to .N suffix"""
        calc_dir = context.source_dir

        # Determine version number
        version = self._get_next_version(calc_dir)

        # Backup all files
        backed_up = []
        failed = []

        for fname in self.INPUT_FILES + self.OUTPUT_FILES:
            src = calc_dir / fname
            dst = calc_dir / f"{fname}.{version}"

            if not src.exists():
                LOGGER.debug(f"File {fname} not found, skipping")
                continue

            try:
                src.rename(dst)
                backed_up.append(f"{fname} → {fname}.{version}")
                LOGGER.debug(f"Backed up {fname} → {fname}.{version}")
            except Exception as e:
                failed.append(f"{fname}: {e}")
                LOGGER.error(f"Failed to backup {fname}: {e}")

        if not backed_up:
            return StageResult(
                success=False,
                target_dir=calc_dir,
                error_message="No files were backed up",
            )

        LOGGER.info(f"✓ Backed up run {version}: {len(backed_up)} files")

        return StageResult(
            success=True,
            target_dir=calc_dir,
            metadata={
                "version": version,
                "backed_up": backed_up,
                "failed": failed,
            },
        )

    def requires_job_completion(self) -> bool:
        return False

    def _get_next_version(self, calc_dir: Path) -> int:
        """Scan for highest .N version and return next"""
        max_version = -1
        for file in calc_dir.glob("OUTCAR.*"):
            try:
                version = int(file.suffix[1:])
                max_version = max(max_version, version)
            except ValueError:
                pass
        return max_version + 1


# ============================================================================
# MD Prepare Stage
# ============================================================================


class MDPrepareStage(Stage):
    """Prepares files for next MD run"""

    INPUT_FILES = ["INCAR", "POSCAR"]

    def __init__(self, name: str = "md_prepare"):
        super().__init__(name)

    def execute(self, context: StageContext) -> StageResult:
        """Restore versioned inputs and geometry for next run"""
        calc_dir = context.source_dir
        current_run = context.global_config.get("current_run", 0)

        try:
            # Restore input files from previous version
            if current_run > 0:
                version = current_run - 1
                for fname in self.INPUT_FILES:
                    src = calc_dir / f"{fname}.{version}"
                    dst = calc_dir / fname
                    if src.exists():
                        dst.write_bytes(src.read_bytes())
                        LOGGER.debug(f"Restored {fname} from version {version}")

                # Get geometry from previous CONTCAR
                src = calc_dir / f"CONTCAR.{version}"
                dst = calc_dir / "POSCAR"
                if src.exists():
                    dst.write_bytes(src.read_bytes())
                    LOGGER.debug(f"Updated POSCAR from CONTCAR.{version}")
                else:
                    LOGGER.warning(f"CONTCAR.{version} not found for geometry update")
            else:
                # First run: just verify POSCAR exists
                if not (calc_dir / "POSCAR").exists():
                    return StageResult(
                        success=False,
                        target_dir=calc_dir,
                        error_message="POSCAR not found for initial run",
                    )

            LOGGER.info(f"✓ Prepared for run {current_run}")

            return StageResult(
                success=True,
                target_dir=calc_dir,
                metadata={"prepared_for_run": current_run},
            )

        except Exception as e:
            LOGGER.error(f"Failed to prepare for run {current_run}: {e}")
            return StageResult(
                success=False,
                target_dir=calc_dir,
                error_message=f"Preparation failed: {e}",
            )

    def requires_job_completion(self) -> bool:
        return False


# ============================================================================
# Workflow Factory
# ============================================================================


def create_md_looping_workflow(
    max_iterations: int | None = None,
    max_duration_hours: float | None = None,
) -> "LoopingWorkflow":
    """Create MD looping workflow with graceful shutdown"""

    return LoopingWorkflow(
        name="md_loop",
        description="Iterative MD run with automatic backup and restart",
        stages=[
            MDSubmitStage("submit"),
            MDValidationStage("validate"),
            MDBackupStage("backup"),
            MDPrepareStage("prepare"),
        ],
        max_iterations=max_iterations,
        max_duration_hours=max_duration_hours,
    )
