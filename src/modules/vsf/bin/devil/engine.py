"""
engine.py
The main loop, state persistence, and SLURM polling
all live together. Timer tracking fits here.
"""

import getpass
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .core import StageContext, StageState, Workflow, WorkflowInstance, WorkflowState

LOGGER = logging.getLogger(__name__)


# ============================================================================
# SLURM Interface
# ============================================================================


class SlurmInterface:
    """All SLURM interactions"""

    def __init__(self, submit_script: str = "vsf-submit-job.py"):
        """
        Args:
            submit_script: Path or name of job submission script
                          Defaults to "vsf-submit-job.py" for production
                          Tests can pass path to mock script
        """
        self.submit_script = submit_script

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
            LOGGER.error(f"squeue failed: {e}")
            return None
        except subprocess.TimeoutExpired:
            LOGGER.error("squeue timed out")
            return None
        except Exception as e:
            LOGGER.error(f"Unexpected error in squeue: {e}")
            return None

    def get_job_count(self) -> Optional[int]:
        """Get count of running jobs for current user"""
        job_ids = self.get_current_job_ids()
        return len(job_ids) if job_ids is not None else None

    def submit_job(self, calc_dir: Path, config: Any) -> Optional[str]:
        """Submit VASP job and return job_id

        Args:
            calc_dir: Calculation directory
            config: MDConfig or SubmitStageConfig with vasp_setup, ntasks, ncore, kpar, algo

        Returns:
            Job ID as string, or None if submission failed
        """
        try:
            cmd = [
                self.submit_script,  # Use instance variable
                str(calc_dir),
                str(config.ntasks),
                str(config.ncore),
                str(config.kpar),
                config.algo,
                "--vasp-setup",
                config.vasp_setup,
                "--nodelist",
                config.nodelist,
            ]

            LOGGER.debug(f"Submitting: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=30
            )

            job_id = self._parse_job_id(result.stdout)
            if job_id:
                LOGGER.info(f"✓ Submitted job {job_id}")
                return job_id
            else:
                LOGGER.warning(f"Could not parse job ID from: {result.stdout}")
                return None

        except Exception as e:
            LOGGER.error(f"Job submission failed: {e}")
            return None

    def is_job_running(self, job_id: str) -> bool:
        """Check if specific job is in queue"""
        current_jobs = self.get_current_job_ids()
        if current_jobs is None:
            return False
        return job_id in current_jobs

    @staticmethod
    def _parse_job_id(output: str) -> Optional[str]:
        """Parse job ID from vsf-submit-job.py output"""
        match = re.search(r"with ID (\d+)", output)
        return match.group(1) if match else None


# ============================================================================
# State Manager
# ============================================================================


class StateManager:
    """Manages state persistence"""

    def __init__(self, state_file: Path):
        self.state_file = state_file

    def load(self) -> Optional[WorkflowState]:
        """Load state from file"""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file) as f:
                return WorkflowState.from_dict(json.load(f))
        except Exception as e:
            LOGGER.error(f"Failed to load state: {e}")
            return None

    def save(self, state: WorkflowState) -> bool:
        """Atomically save state"""
        try:
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            temp_file.replace(self.state_file)
            return True
        except Exception as e:
            LOGGER.error(f"Failed to save state: {e}")
            return False

    def create_initial_state(
        self, config: Dict, workflow: Workflow, directories: List[Path]
    ) -> WorkflowState:
        """Create initial state for new run"""
        workflow_instances = {}

        for directory in directories:
            stages = [
                StageState(
                    name=stage.name,
                    status="pending",
                    target_dir=None,
                    job_id=None,
                )
                for stage in workflow.stages
            ]

            workflow_instances[str(directory)] = WorkflowInstance(
                workflow_name=workflow.name,
                status="pending",
                current_stage=0,
                stages=stages,
            )

        return WorkflowState(
            config=config,
            workflow_name=workflow.name,
            created_at=datetime.now().isoformat(),
            workflow_instances=workflow_instances,
        )


# ============================================================================
# Workflow Engine
# ============================================================================


class WorkflowEngine:
    """Executes workflow stages"""

    def __init__(self, slurm: SlurmInterface):
        self.slurm = slurm

    def execute_next_stage(
        self,
        workflow_instance: WorkflowInstance,
        workflow_def: Workflow,
        source_dir: Path,
        global_config: Dict,
    ) -> bool:
        """
        Try to execute the next pending stage.
        Returns True if a stage was executed.
        """
        stage_idx = workflow_instance.current_stage

        if stage_idx >= len(workflow_def.stages):
            workflow_instance.status = "completed"
            return False

        stage_def = workflow_def.stages[stage_idx]
        stage_state = workflow_instance.stages[stage_idx]

        if stage_state.status != "pending":
            return False

        # Build context from previous stage
        prev_idx = stage_idx - 1
        prev_target_dir = None
        prev_job_id = None

        if prev_idx >= 0:
            prev_state = workflow_instance.stages[prev_idx]
            prev_target_dir = prev_state.target_dir
            prev_job_id = prev_state.job_id

        context = StageContext(
            source_dir=source_dir,
            workflow_name=workflow_instance.workflow_name,
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
            LOGGER.info(f"Executing stage '{stage_def.name}' for {source_dir}")

            result = stage_def.execute(context)
            stage_def.validate_result(result)

            if result.success:
                # Update state based on stage type
                if stage_def.requires_job_completion():
                    stage_state.status = "running"
                    stage_state.job_id = result.job_id
                    workflow_instance.status = "in_progress"
                else:
                    stage_state.status = "completed"
                    workflow_instance.current_stage += 1

                stage_state.target_dir = result.target_dir
                stage_state.executed_at = datetime.now().isoformat()

                # Log metadata
                if result.metadata:
                    LOGGER.debug(
                        f"Stage '{stage_def.name}' metadata: {result.metadata}"
                    )

                return True
            else:
                stage_state.status = "failed"
                stage_state.error = result.error_message
                workflow_instance.status = "failed"
                LOGGER.error(f"Stage '{stage_def.name}' failed: {result.error_message}")
                return False

        except Exception as e:
            stage_state.status = "failed"
            stage_state.error = str(e)
            workflow_instance.status = "failed"
            LOGGER.exception(f"Exception in stage '{stage_def.name}'")
            return False

    def update_running_stages(
        self, workflow_instances: Dict[str, WorkflowInstance], workflow_def: Workflow
    ):
        """Check status of running jobs and advance completed stages"""
        current_jobs = self.slurm.get_current_job_ids()

        if current_jobs is None:
            LOGGER.warning("Could not get current jobs, skipping update")
            return

        for source_dir_str, instance in workflow_instances.items():
            if instance.status not in ["pending", "in_progress"]:
                continue

            stage_idx = instance.current_stage
            if stage_idx >= len(instance.stages):
                continue

            stage_state = instance.stages[stage_idx]

            if stage_state.status == "running":
                job_id = stage_state.job_id
                if job_id and job_id not in current_jobs:
                    # Job completed
                    stage_state.status = "completed"
                    stage_state.completed_at = datetime.now().isoformat()
                    instance.current_stage += 1

                    LOGGER.info(
                        f"Job {job_id} completed for {source_dir_str}, "
                        f"stage '{stage_state.name}'"
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
            LOGGER.info("Creating new state")
            self.state = self.state_manager.create_initial_state(
                config, workflow, directories
            )
            self.state_manager.save(self.state)
        else:
            self.state = state
            LOGGER.info(
                f"Loaded existing state with {len(self.state.workflow_instances)} workflows"
            )

    def run(self, dry_run: bool = False):
        """Main execution loop"""
        if dry_run:
            self._dry_run()
            return

        LOGGER.info("=" * 60)
        LOGGER.info(f"VASP Devil starting: {self.workflow.description}")
        LOGGER.info(f"Max jobs: {self.config['max_jobs']}")
        LOGGER.info(f"Workflow instances: {len(self.state.workflow_instances)}")
        LOGGER.info("=" * 60)

        try:
            iteration = 0
            while True:
                iteration += 1
                LOGGER.info(f"\n--- Iteration {iteration} ---")

                # Update status of running jobs
                self.engine.update_running_stages(
                    self.state.workflow_instances, self.workflow
                )

                # Check if all workflows complete
                if self._all_workflows_complete():
                    LOGGER.info("\n" + "=" * 60)
                    LOGGER.info("All workflows completed!")
                    self._print_summary()
                    LOGGER.info("=" * 60)
                    break

                # Count active jobs
                active_jobs = self._count_active_jobs()
                available_slots = self.config["max_jobs"] - active_jobs

                LOGGER.info(
                    f"Active jobs: {active_jobs}/{self.config['max_jobs']}, "
                    f"Available slots: {available_slots}"
                )

                # Try to submit new stages
                if available_slots > 0:
                    submitted = self._submit_next_stages(available_slots)
                    LOGGER.info(f"Submitted {submitted} new stages")

                # Save state
                self.state_manager.save(self.state)

                # Sleep before next iteration
                LOGGER.debug(f"Sleeping for {self.sleep_time}s...")
                time.sleep(self.sleep_time)

        except KeyboardInterrupt:
            LOGGER.info("\nInterrupted by user. Saving state...")
            self.state_manager.save(self.state)
            sys.exit(0)

    def _dry_run(self):
        """Show what would be executed"""
        LOGGER.info("DRY RUN MODE")
        LOGGER.info(f"Workflow: {self.workflow.name} - {self.workflow.description}")
        LOGGER.info(f"\nStages:")
        for i, stage in enumerate(self.workflow.stages):
            job_marker = " [SUBMITS JOB]" if stage.requires_job_completion() else ""
            LOGGER.info(f"  {i+1}. {stage.name}{job_marker}")

        LOGGER.info(
            f"\nWould process {len(self.state.workflow_instances)} directories:"
        )
        for dir_path in self.state.workflow_instances.keys():
            LOGGER.info(f"  - {dir_path}")

    def _count_active_jobs(self) -> int:
        """Count currently active jobs across all workflows"""
        count = 0
        for instance in self.state.workflow_instances.values():
            for stage in instance.stages:
                if stage.status == "running" and stage.job_id:
                    count += 1
        return count

    def _submit_next_stages(self, available_slots: int) -> int:
        """Submit next stages for workflows with available capacity"""
        submitted = 0

        for source_dir_str, instance in self.state.workflow_instances.items():
            if submitted >= available_slots:
                break

            if instance.status == "failed":
                continue

            source_dir = Path(source_dir_str)

            executed = self.engine.execute_next_stage(
                instance, self.workflow, source_dir, self.config
            )

            if executed:
                # Check if it submitted a job (counts against limit)
                stage_idx = instance.current_stage
                if stage_idx > 0:  # Stage was executed, current_stage advanced
                    prev_stage = instance.stages[stage_idx - 1]
                    if prev_stage.job_id:
                        submitted += 1

        return submitted

    def _all_workflows_complete(self) -> bool:
        """Check if all workflows are in terminal state"""
        for instance in self.state.workflow_instances.values():
            if instance.status not in ["completed", "failed"]:
                return False
        return True

    def _print_summary(self):
        """Print summary of all workflows"""
        completed = sum(
            1
            for inst in self.state.workflow_instances.values()
            if inst.status == "completed"
        )
        failed = sum(
            1
            for inst in self.state.workflow_instances.values()
            if inst.status == "failed"
        )

        LOGGER.info(f"Summary:")
        LOGGER.info(f"  Completed: {completed}")
        LOGGER.info(f"  Failed: {failed}")

        if failed > 0:
            LOGGER.info("\nFailed workflows:")
            for dir_path, inst in self.state.workflow_instances.items():
                if inst.status == "failed":
                    failed_stage = next(s for s in inst.stages if s.status == "failed")
                    LOGGER.info(
                        f"  {dir_path}: stage '{failed_stage.name}' - "
                        f"{failed_stage.error or 'unknown error'}"
                    )
