#!/usr/bin/env python3
"""
VASP Devil - Prototype Implementation
Tests the workflow architecture with a complete double relaxation workflow.
"""

import getpass
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

LOGGER = logging.getLogger(__name__)

# ============================================================================
# Core Data Structures
# ============================================================================


@dataclass(frozen=True)
class StageContext:
    """
    Immutable context passed to every stage.
    Contains only essential information.
    """

    source_dir: Path
    workflow_name: str
    stage_index: int

    # Previous stage outputs
    previous_target_dir: Optional[Path] = None
    previous_job_id: Optional[str] = None

    # Configs (read-only)
    global_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """
    Result of executing a stage.
    This is the ONLY way stages communicate forward.
    """

    success: bool
    target_dir: Path
    job_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.success and not self.error_message:
            raise ValueError("Failed results must have error_message")


# ============================================================================
# Directory Resolution
# ============================================================================


class DirectoryResolver:
    """Resolves directory expressions with variable substitution"""

    @staticmethod
    def resolve(expr: str, context: StageContext) -> Path:
        """
        Resolve directory expression.

        Available variables:
        - {source_dir}: Original workflow directory
        - {prev_dir}: Previous stage's target directory
        """
        replacements = {
            "source_dir": str(context.source_dir),
        }

        if context.previous_target_dir:
            replacements["prev_dir"] = str(context.previous_target_dir)

        resolved = expr.format(**replacements)
        return Path(resolved)


# ============================================================================
# Stage Base Class
# ============================================================================


class Stage(ABC):
    """Base class for all workflow stages"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, context: StageContext) -> StageResult:
        """Execute this stage"""
        pass

    @abstractmethod
    def requires_job_completion(self) -> bool:
        """Does this stage submit a job that must complete?"""
        pass

    def can_execute(self, context: StageContext) -> bool:
        """Check if prerequisites are met (override if needed)"""
        return True

    def validate_result(self, result: StageResult) -> None:
        """Validate stage result"""
        if result.success:
            if not result.target_dir:
                raise ValueError(
                    f"Stage {self.name}: successful result must have target_dir"
                )
            if self.requires_job_completion() and not result.job_id:
                raise ValueError(
                    f"Stage {self.name}: job-submitting stage must return job_id"
                )


# TODO: write a stage that modfies the INCAR input between the relaxations.
# discuss with chat entry point, but I believe this is just a new stage in the flow

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
            logging.info(f"Submitting job for {target_dir}")
            logging.debug(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=30
            )

            # Parse job ID from output
            job_id = self._parse_job_id(result.stdout)

            if not job_id:
                logging.warning(f"Could not parse job ID from output:\n{result.stdout}")
                return StageResult(
                    success=False,
                    target_dir=target_dir,
                    error_message="Could not parse job ID from submission output",
                )

            logging.info(f"✓ Submitted job {job_id} for {target_dir}")

            return StageResult(
                success=True,
                target_dir=target_dir,
                job_id=job_id,
                metadata={"submit_output": result.stdout.strip()},
            )

        except subprocess.CalledProcessError as e:
            logging.error(f"Submission failed: {e.stderr}")
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
            logging.exception(f"Unexpected error during submission")
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
            logging.error(f"Missing required files in {path}: {missing}")
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
                logging.info(f"Created directory: {target_dir}")
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
                logging.warning(f"File {src_file} not found in {source_dir}, skipping")
                continue

            dst_name = self.config.rename_map.get(src_file, src_file)
            dst_path = target_dir / dst_name

            try:
                shutil.copy2(src_path, dst_path)
                copied_files.append(f"{src_file} → {dst_name}")
                logging.debug(f"Copied {src_file} → {dst_path}")
            except Exception as e:
                failed_files.append(f"{src_file}: {e}")
                logging.error(f"Failed to copy {src_file}: {e}")

        # Success if we copied at least one file
        success = len(copied_files) > 0

        if success:
            logging.info(
                f"✓ Copied {len(copied_files)} files: {source_dir} → {target_dir}"
            )

        if missing_files:
            logging.info(f"Missing files (skipped): {missing_files}")

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
                logging.info(
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
# Workflow Definition
# ============================================================================


@dataclass
class Workflow:
    """A workflow is an ordered sequence of stages"""

    name: str
    stages: List[Stage]
    description: str = ""

    def validate(self):
        """Validate workflow structure"""
        if not self.stages:
            raise ValueError("Workflow must have at least one stage")

        names = [s.name for s in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Stage names must be unique")


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


# ============================================================================
# SLURM Interface
# ============================================================================


class SlurmInterface:
    """All SLURM interactions"""

    def get_current_job_ids(self) -> Optional[Set[str]]:
        """Get set of running job IDs for current user"""
        try:
            result = subprocess.run(
                ["squeue", "-u", getpass.getuser(), "-h", "-o", "%i"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            output = result.stdout.strip()
            return set(output.split("\n")) if output else set()
        except subprocess.CalledProcessError as e:
            logging.error(f"squeue failed: {e}")
            return None
        except subprocess.TimeoutExpired:
            logging.error("squeue timed out")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in squeue: {e}")
            return None

    def get_job_count(self) -> Optional[int]:
        """Get count of running jobs for current user"""
        job_ids = self.get_current_job_ids()
        return len(job_ids) if job_ids is not None else None


# ============================================================================
# State Manager
# ============================================================================


class StateManager:
    """Manages state persistence"""

    def __init__(self, state_file: Path):
        self.state_file = state_file

    def load(self) -> Optional[Dict]:
        """Load state from file"""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file) as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load state: {e}")
            return None

    def save(self, state: Dict) -> bool:
        """Atomically save state"""
        try:
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
            temp_file.replace(self.state_file)
            return True
        except Exception as e:
            logging.error(f"Failed to save state: {e}")
            return False

    def create_initial_state(
        self, config: Dict, workflow: Workflow, directories: List[Path]
    ) -> Dict:
        """Create initial state for new run"""
        state = {
            "config": config,
            "workflow_name": workflow.name,
            "created_at": datetime.now().isoformat(),
            "workflow_instances": {},
        }

        for directory in directories:
            state["workflow_instances"][str(directory)] = {
                "workflow_name": workflow.name,
                "status": "pending",
                "current_stage": 0,
                "stages": [
                    {
                        "name": stage.name,
                        "status": "pending",
                        "target_dir": None,
                        "job_id": None,
                    }
                    for stage in workflow.stages
                ],
            }

        return state


# ============================================================================
# Workflow Engine
# ============================================================================


class WorkflowEngine:
    """Executes workflow stages"""

    def __init__(self, slurm: SlurmInterface):
        self.slurm = slurm

    def execute_next_stage(
        self,
        workflow_instance: Dict,
        workflow_def: Workflow,
        source_dir: Path,
        global_config: Dict,
    ) -> bool:
        """
        Try to execute the next pending stage.
        Returns True if a stage was executed.
        """
        stage_idx = workflow_instance["current_stage"]

        if stage_idx >= len(workflow_def.stages):
            workflow_instance["status"] = "completed"
            return False

        stage_def = workflow_def.stages[stage_idx]
        stage_state = workflow_instance["stages"][stage_idx]

        if stage_state["status"] != "pending":
            return False

        # Build context from previous stage
        prev_idx = stage_idx - 1
        prev_target_dir = None
        prev_job_id = None

        if prev_idx >= 0:
            prev_state = workflow_instance["stages"][prev_idx]
            prev_target_dir = (
                Path(prev_state["target_dir"]) if prev_state.get("target_dir") else None
            )
            prev_job_id = prev_state.get("job_id")

        context = StageContext(
            source_dir=source_dir,
            workflow_name=workflow_instance["workflow_name"],
            stage_index=stage_idx,
            previous_target_dir=prev_target_dir,
            previous_job_id=prev_job_id,
            global_config=global_config,
        )

        # Check prerequisites
        if not stage_def.can_execute(context):
            return False

        # Execute stage
        try:
            logging.info(f"Executing stage '{stage_def.name}' for {source_dir}")

            result = stage_def.execute(context)
            stage_def.validate_result(result)

            if result.success:
                # Update state based on stage type
                if stage_def.requires_job_completion():
                    stage_state["status"] = "running"
                    stage_state["job_id"] = result.job_id
                    workflow_instance["status"] = "in_progress"
                else:
                    stage_state["status"] = "completed"
                    workflow_instance["current_stage"] += 1

                stage_state["target_dir"] = str(result.target_dir)
                stage_state["executed_at"] = datetime.now().isoformat()

                # Log metadata
                if result.metadata:
                    logging.debug(
                        f"Stage '{stage_def.name}' metadata: {result.metadata}"
                    )

                return True
            else:
                stage_state["status"] = "failed"
                stage_state["error"] = result.error_message
                workflow_instance["status"] = "failed"
                logging.error(
                    f"Stage '{stage_def.name}' failed: {result.error_message}"
                )
                return False

        except Exception as e:
            stage_state["status"] = "failed"
            stage_state["error"] = str(e)
            workflow_instance["status"] = "failed"
            logging.exception(f"Exception in stage '{stage_def.name}'")
            return False

    def update_running_stages(self, workflow_instances: Dict, workflow_def: Workflow):
        """Check status of running jobs and advance completed stages"""
        current_jobs = self.slurm.get_current_job_ids()

        if current_jobs is None:
            logging.warning("Could not get current jobs, skipping update")
            return

        for source_dir_str, instance in workflow_instances.items():
            if instance["status"] not in ["pending", "in_progress"]:
                continue

            stage_idx = instance["current_stage"]
            if stage_idx >= len(instance["stages"]):
                continue

            stage_state = instance["stages"][stage_idx]

            if stage_state["status"] == "running":
                job_id = stage_state.get("job_id")
                if job_id and job_id not in current_jobs:
                    # Job completed
                    stage_state["status"] = "completed"
                    stage_state["completed_at"] = datetime.now().isoformat()
                    instance["current_stage"] += 1

                    logging.info(
                        f"Job {job_id} completed for {source_dir_str}, "
                        f"stage '{stage_state['name']}'"
                    )


# ============================================================================
# Main Orchestrator
# ============================================================================


class VASPDevil:
    """Main orchestrator"""

    def __init__(
        self,
        workflow: Workflow,
        config: Dict,
        directories: List[Path],
        state_file: Path,
        sleep_time: float = 30,
    ):
        self.workflow = workflow
        self.config = config
        self.sleep_time = sleep_time

        self.state_manager = StateManager(state_file)
        self.slurm = SlurmInterface()
        self.engine = WorkflowEngine(self.slurm)

        # Load or create state
        state = self.state_manager.load()

        if state is None:
            logging.info("Creating new state")
            self.state = self.state_manager.create_initial_state(
                config, workflow, directories
            )
            self.state_manager.save(self.state)
        else:
            self.state = state
            logging.info(
                f"Loaded existing state with {len(self.state['workflow_instances'])} workflows"
            )

    def run(self, dry_run: bool = False):
        """Main execution loop"""
        if dry_run:
            self._dry_run()
            return

        logging.info("=" * 60)
        logging.info(f"VASP Devil starting: {self.workflow.description}")
        logging.info(f"Max jobs: {self.config['max_jobs']}")
        logging.info(f"Workflow instances: {len(self.state['workflow_instances'])}")
        logging.info("=" * 60)

        try:
            iteration = 0
            while True:
                iteration += 1
                logging.info(f"\n--- Iteration {iteration} ---")

                # Update status of running jobs
                self.engine.update_running_stages(
                    self.state["workflow_instances"], self.workflow
                )

                # Check if all workflows complete
                if self._all_workflows_complete():
                    logging.info("\n" + "=" * 60)
                    logging.info("All workflows completed!")
                    self._print_summary()
                    logging.info("=" * 60)
                    break

                # Count active jobs
                active_jobs = self._count_active_jobs()
                available_slots = self.config["max_jobs"] - active_jobs

                logging.info(
                    f"Active jobs: {active_jobs}/{self.config['max_jobs']}, "
                    f"Available slots: {available_slots}"
                )

                # Try to submit new stages
                if available_slots > 0:
                    submitted = self._submit_next_stages(available_slots)
                    logging.info(f"Submitted {submitted} new stages")

                # Save state
                self.state_manager.save(self.state)

                # Sleep before next iteration
                logging.debug(f"Sleeping for {self.sleep_time}s...")
                time.sleep(self.sleep_time)

        except KeyboardInterrupt:
            logging.info("\nInterrupted by user. Saving state...")
            self.state_manager.save(self.state)
            sys.exit(0)

    def _dry_run(self):
        """Show what would be executed"""
        logging.info("DRY RUN MODE")
        logging.info(f"Workflow: {self.workflow.name} - {self.workflow.description}")
        logging.info(f"\nStages:")
        for i, stage in enumerate(self.workflow.stages):
            job_marker = " [SUBMITS JOB]" if stage.requires_job_completion() else ""
            logging.info(f"  {i+1}. {stage.name}{job_marker}")

        logging.info(
            f"\nWould process {len(self.state['workflow_instances'])} directories:"
        )
        for dir_path in self.state["workflow_instances"].keys():
            logging.info(f"  - {dir_path}")

    def _count_active_jobs(self) -> int:
        """Count currently active jobs across all workflows"""
        count = 0
        for instance in self.state["workflow_instances"].values():
            for stage in instance["stages"]:
                if stage["status"] == "running" and stage.get("job_id"):
                    count += 1
        return count

    def _submit_next_stages(self, available_slots: int) -> int:
        """Submit next stages for workflows with available capacity"""
        submitted = 0

        for source_dir_str, instance in self.state["workflow_instances"].items():
            if submitted >= available_slots:
                break

            if instance["status"] == "failed":
                continue

            source_dir = Path(source_dir_str)

            executed = self.engine.execute_next_stage(
                instance, self.workflow, source_dir, self.config
            )

            if executed:
                # Check if it submitted a job (counts against limit)
                stage_idx = instance["current_stage"]
                if stage_idx > 0:  # Stage was executed, current_stage advanced
                    prev_stage = instance["stages"][stage_idx - 1]
                    if prev_stage.get("job_id"):
                        submitted += 1

        return submitted

    def _all_workflows_complete(self) -> bool:
        """Check if all workflows are in terminal state"""
        for instance in self.state["workflow_instances"].values():
            if instance["status"] not in ["completed", "failed"]:
                return False
        return True

    def _print_summary(self):
        """Print summary of all workflows"""
        completed = sum(
            1
            for inst in self.state["workflow_instances"].values()
            if inst["status"] == "completed"
        )
        failed = sum(
            1
            for inst in self.state["workflow_instances"].values()
            if inst["status"] == "failed"
        )

        logging.info(f"Summary:")
        logging.info(f"  Completed: {completed}")
        logging.info(f"  Failed: {failed}")

        if failed > 0:
            logging.info("\nFailed workflows:")
            for dir_path, inst in self.state["workflow_instances"].items():
                if inst["status"] == "failed":
                    failed_stage = next(
                        s for s in inst["stages"] if s["status"] == "failed"
                    )
                    logging.info(
                        f"  {dir_path}: stage '{failed_stage['name']}' - "
                        f"{failed_stage.get('error', 'unknown error')}"
                    )


# ============================================================================
# CLI and Main
# ============================================================================


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"vasp_devil_{timestamp}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
    )

    logging.info(f"Log file: {log_file}")


def find_directories(glob_pattern: str) -> List[Path]:
    """Find directories matching glob pattern"""
    glob_path = Path(glob_pattern)

    if glob_path.is_absolute():
        pattern_parent = glob_path.parent
        pattern_name = glob_path.name
    else:
        pattern_parent = Path.cwd()
        pattern_name = glob_pattern

    try:
        if any(c in pattern_name for c in ["*", "?", "["]):
            directories = [d for d in pattern_parent.glob(pattern_name) if d.is_dir()]
        else:
            single_dir = pattern_parent / pattern_name
            directories = [single_dir] if single_dir.is_dir() else []
    except Exception as e:
        logging.error(f"Error processing glob pattern '{glob_pattern}': {e}")
        return []

    return directories


def main():
    """Main entry point for testing"""
    import argparse

    parser = argparse.ArgumentParser(
        description="VASP Devil - Workflow Prototype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("glob_pattern", help="Glob pattern for directories")
    parser.add_argument("ntasks", type=int, help="Number of MPI tasks")
    parser.add_argument("ncore", type=int, help="VASP NCORE")
    parser.add_argument("kpar", type=int, help="VASP KPAR")
    parser.add_argument("algo", help="VASP ALGO")
    parser.add_argument("--vasp-setup", required=True, help="VASP setup command")
    parser.add_argument("--max-jobs", type=int, default=10, help="Max concurrent jobs")
    parser.add_argument("--workflow", choices=["simple", "double"], default="simple")
    parser.add_argument("--partition", help="SLURM partition")
    parser.add_argument("--state-file", default="vasp-devil-state.json")
    parser.add_argument("--sleep-time", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Validate parallelization
    if args.ntasks % (args.ncore * args.kpar) != 0:
        logging.error(
            f"Invalid parallelization: ntasks ({args.ntasks}) must be "
            f"divisible by (ncore × kpar) = {args.ncore * args.kpar}"
        )
        sys.exit(1)

    # Find directories
    directories = find_directories(args.glob_pattern)
    if not directories:
        logging.error(f"No directories found matching: {args.glob_pattern}")
        sys.exit(1)

    logging.info(f"Found {len(directories)} directories")

    # Create workflow
    slurm_kwargs = {}
    if args.partition:
        slurm_kwargs["partition"] = args.partition

    if args.workflow == "simple":
        workflow = create_simple_workflow(
            args.ntasks, args.ncore, args.kpar, args.algo, **slurm_kwargs
        )
    else:
        workflow = create_double_relaxation_workflow(
            args.ntasks, args.ncore, args.kpar, args.algo, **slurm_kwargs
        )

    workflow.validate()

    # Create config
    config = {
        "max_jobs": args.max_jobs,
        "vasp_setup": args.vasp_setup,
        "ntasks": args.ntasks,
        "ncore": args.ncore,
        "kpar": args.kpar,
        "algo": args.algo,
    }

    # Create and run devil
    devil = VASPDevil(
        workflow=workflow,
        config=config,
        directories=directories,
        state_file=Path(args.state_file),
        sleep_time=args.sleep_time,
    )

    devil.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
