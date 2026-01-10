"""
md_runner.py
MD Orchestrator - Manages iterative MD runs with graceful shutdown.
Handles backup versioning, recovery, and timer/iteration-based loop control.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class MDConfig:
    """Configuration for MD orchestration"""

    vasp_setup: str
    ntasks: int = 128
    ncore: int = 32
    kpar: int = 4
    algo: str = "Normal"
    nodelist: str = ""
    # "ct01,ct02,ct03,ct04,ct06,ct08,ct09,ct10,gn01,gn02,gn03,gn04,gn05,gn06,gn07,gn08,gn09,gn10,gn11,gn12,gn13,gn14,gn15,gn16,gn17,gn18,gn18,gn19,gn20,gn21"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MDConfig":
        return cls(**data)


@dataclass
class MDStats:
    """Statistics tracked during MD orchestration"""

    completed_runs: int = 0
    failed_runs: int = 0
    run_durations: Dict[int, float] = field(default_factory=dict)  # {run_num: seconds}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MDStats":
        return cls(
            completed_runs=data.get("completed_runs", 0),
            failed_runs=data.get("failed_runs", 0),
            run_durations={int(k): v for k, v in data.get("run_durations", {}).items()},
        )


@dataclass
class MDRunMetadata:
    """Metadata for a single MD run"""

    run_number: int
    job_id: Optional[str] = None
    submitted_at: Optional[str] = None  # ISO format
    completed_at: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    elapsed_time_sec: Optional[float] = None


@dataclass
class MDState:
    """Persistent state for MD orchestration"""

    calc_dir: Path
    config: MDConfig
    stats: MDStats
    current_run: int
    start_time: str  # ISO format
    max_iterations: Optional[int]
    max_duration_hours: Optional[float]
    status: str  # pending, running, completed, stopped
    runs: Dict[int, MDRunMetadata] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calc_dir": str(self.calc_dir),
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
            "current_run": self.current_run,
            "start_time": self.start_time,
            "max_iterations": self.max_iterations,
            "max_duration_hours": self.max_duration_hours,
            "status": self.status,
            "runs": {k: asdict(v) for k, v in self.runs.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MDState":
        return cls(
            calc_dir=Path(data["calc_dir"]),
            config=MDConfig.from_dict(data["config"]),
            stats=MDStats.from_dict(data.get("stats", {})),
            current_run=data["current_run"],
            start_time=data["start_time"],
            max_iterations=data.get("max_iterations"),
            max_duration_hours=data.get("max_duration_hours"),
            status=data["status"],
            runs={int(k): MDRunMetadata(**v) for k, v in data.get("runs", {}).items()},
        )


# ============================================================================
# Backup Manager
# ============================================================================


class MDBackupManager:
    """Manages file versioning for MD runs"""

    INPUT_FILES = ["INCAR", "KPOINTS", "POTCAR"]

    # All other output files that VASP creates/overwrites
    OUTPUT_FILES = [
        "CHG",
        "CHGCAR",
        "DOSCAR",
        "EIGENVAL",
        "IBZKPT",
        "ICONST",
        "OSZICAR",
        "OUTCAR",
        "PCDAT",
        "REPORT",
        "WAVECAR",
        "vaspout.h5",
        "vasprun.xml",
        "XDATCAR",
    ]

    def __init__(self, calc_dir: Path):
        self.calc_dir = calc_dir
        self.bak_dir = calc_dir / "bak"
        self.bak_dir.mkdir(exist_ok=True)

    def archive_run(self, version: int) -> bool:
        """After run completes: move all outputs and POSCAR to bak.{version}/ directory"""
        try:
            run_backup_dir = self.bak_dir / f"run.{version}"
            run_backup_dir.mkdir(exist_ok=True)

            # Archive POSCAR used for this run
            poscar = self.calc_dir / "POSCAR"
            if poscar.exists():
                poscar.rename(run_backup_dir / "POSCAR")

            # Archive CONTCAR output
            contcar = self.calc_dir / "CONTCAR"
            if contcar.exists():
                contcar.rename(run_backup_dir / "CONTCAR")

            # Archive all output files
            for fname in self.OUTPUT_FILES:
                src = self.calc_dir / fname
                if src.exists():
                    src.rename(run_backup_dir / fname)

            LOGGER.info(f"✓ Archived run {version} to bak/run.{version}/")
            return True

        except Exception as e:
            LOGGER.error(f"Failed to archive run {version}: {e}")
            return False

    def prepare_geometry_for_next_run(self, run_version: int) -> bool:
        """Before submitting: restore POSCAR for next run from previous CONTCAR"""
        try:
            poscar = self.calc_dir / "POSCAR"

            if run_version == 0:
                # First run - POSCAR should already exist
                if not poscar.exists():
                    LOGGER.error("POSCAR not found for initial run")
                    return False
                LOGGER.debug("Using initial POSCAR for run 0")
                return True

            # Continuation - use CONTCAR from previous run
            prev_contcar = self.bak_dir / f"run.{run_version - 1}" / "CONTCAR"
            if not prev_contcar.exists():
                LOGGER.error(f"CONTCAR from run {run_version - 1} not found")
                return False

            poscar.write_bytes(prev_contcar.read_bytes())
            LOGGER.info(f"✓ Prepared POSCAR for run {run_version}")
            return True

        except Exception as e:
            LOGGER.error(f"Failed to prepare geometry: {e}")
            return False


# ============================================================================
# State Manager
# ============================================================================


class MDStateManager:
    """Manages MD state persistence"""

    def __init__(self, state_file: Path):
        self.state_file = state_file

    def load(self) -> Optional[MDState]:
        """Load state from file"""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file) as f:
                return MDState.from_dict(json.load(f))
        except Exception as e:
            LOGGER.error(f"Failed to load state: {e}")
            return None

    def save(self, state: MDState) -> bool:
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
        self,
        calc_dir: Path,
        config: MDConfig,
        max_iterations: Optional[int],
        max_duration_hours: Optional[float],
    ) -> MDState:
        """Create initial state"""
        return MDState(
            calc_dir=calc_dir,
            config=config,
            stats=MDStats(),
            current_run=0,
            start_time=datetime.now().isoformat(),
            max_iterations=max_iterations,
            max_duration_hours=max_duration_hours,
            status="pending",
            runs={},
        )


# ============================================================================
# MD Orchestrator
# ============================================================================


class MDOrchestrator:
    """Main MD orchestration loop"""

    def __init__(
        self,
        calc_dir: Path,
        config: MDConfig,
        state_file: Path,
        max_iterations: Optional[int] = None,
        max_duration_hours: Optional[float] = None,
        sleep_time: float = 30,
        submit_script: str = "vsf-submit-job.py",  # NEW
    ):
        self.calc_dir = Path(calc_dir)
        self.config = config
        self.sleep_time = sleep_time

        self.state_manager = MDStateManager(state_file)
        self.backup_manager = MDBackupManager(self.calc_dir)

        # Import here to avoid circular dependency
        from .engine import SlurmInterface

        self.slurm = SlurmInterface(submit_script=submit_script)  # Pass it through

        # Load or create state
        state = self.state_manager.load()

        if state is None:
            LOGGER.info("Creating new MD state")
            self.state = self.state_manager.create_initial_state(
                calc_dir, config, max_iterations, max_duration_hours
            )
            self.state_manager.save(self.state)
        else:
            self.state = state
            LOGGER.info(
                f"Loaded existing MD state: {self.state.current_run} runs completed"
            )

    def run(self, dry_run: bool = False):
        """Main execution loop"""
        if dry_run:
            self._dry_run()
            return

        LOGGER.info("=" * 60)
        LOGGER.info(f"MD Orchestrator starting: {self.calc_dir.name}")
        if self.state.max_iterations:
            LOGGER.info(f"Max iterations: {self.state.max_iterations}")
        if self.state.max_duration_hours:
            LOGGER.info(f"Max duration: {self.state.max_duration_hours} hours")
        LOGGER.info("=" * 60)

        try:
            iteration = 0
            while True:
                iteration += 1
                LOGGER.info(f"\n--- Iteration {iteration} ---")

                # ===== UPDATE JOB STATUS FIRST =====
                self._check_running_job()

                # ===== CHECK TERMINATION CONDITIONS =====
                elapsed_sec = (
                    datetime.now() - datetime.fromisoformat(self.state.start_time)
                ).total_seconds()
                elapsed_hours = elapsed_sec / 3600

                if self.state.max_duration_hours:
                    remaining_hours = self.state.max_duration_hours - elapsed_hours
                    LOGGER.info(
                        f"Elapsed: {elapsed_hours:.1f}h / {self.state.max_duration_hours}h "
                        f"(remaining: {remaining_hours:.1f}h)"
                    )

                if self.state.max_iterations:
                    LOGGER.info(
                        f"Completed runs: {self.state.stats.completed_runs} / {self.state.max_iterations}"
                    )

                # Check if we should stop
                should_stop = False

                if (
                    self.state.max_duration_hours
                    and elapsed_hours >= self.state.max_duration_hours
                ):
                    LOGGER.info("Max duration reached, stopping...")
                    should_stop = True

                if (
                    self.state.max_iterations
                    and self.state.stats.completed_runs >= self.state.max_iterations
                ):
                    LOGGER.info("Max iterations reached, stopping...")
                    should_stop = True

                if should_stop:
                    self._write_stopcar()
                    self.state.status = "stopped"
                    self.state_manager.save(self.state)
                    self._print_summary()
                    break

                # ===== SUBMIT NEXT JOB (only if not stopping) =====
                if self.state.status in ["pending", "running"]:
                    self._try_submit_next_job()

                # Save state
                self.state_manager.save(self.state)

                # Sleep
                LOGGER.debug(f"Sleeping {self.sleep_time}s...")
                time.sleep(self.sleep_time)

        except KeyboardInterrupt:
            LOGGER.info("\nInterrupted by user, writing STOPCAR...")
            self._write_stopcar()
            self.state_manager.save(self.state)
            self._print_summary()

    def _dry_run(self):
        """Show what would execute"""
        LOGGER.info("DRY RUN MODE")
        LOGGER.info(f"Calculation directory: {self.calc_dir}")
        LOGGER.info(f"Max iterations: {self.state.max_iterations}")
        LOGGER.info(f"Max duration: {self.state.max_duration_hours} hours")
        LOGGER.info(f"Current run: {self.state.current_run}")
        LOGGER.info(f"\nVASP Config:")
        LOGGER.info(f"  ntasks: {self.state.config.ntasks}")
        LOGGER.info(f"  ncore: {self.state.config.ncore}")
        LOGGER.info(f"  kpar: {self.state.config.kpar}")
        LOGGER.info(f"  algo: {self.state.config.algo}")

        if self.state.runs:
            LOGGER.info(f"\nPrevious runs:")
            for run_num, metadata in sorted(self.state.runs.items()):
                LOGGER.info(
                    f"  Run {run_num}: {metadata.status} (job {metadata.job_id})"
                )

    def _check_running_job(self):
        """Check if current job completed"""
        if self.state.status != "running":
            return

        current_run = self.state.current_run - 1
        if current_run not in self.state.runs:
            return

        run_meta = self.state.runs[current_run]
        if not run_meta.job_id:
            return

        # Check if job is still running
        if not self.slurm.is_job_running(run_meta.job_id):
            run_meta.status = "completed"
            run_meta.completed_at = datetime.now().isoformat()

            if run_meta.submitted_at:
                submitted_dt = datetime.fromisoformat(run_meta.submitted_at)
                completed_dt = datetime.fromisoformat(run_meta.completed_at)
                elapsed_sec = (completed_dt - submitted_dt).total_seconds()
                run_meta.elapsed_time_sec = elapsed_sec
                self.state.stats.run_durations[current_run] = elapsed_sec

            self.state.stats.completed_runs += 1
            self.state.status = "pending"

            LOGGER.info(f"✓ Job {run_meta.job_id} completed (run {current_run})")

            # Archive outputs to bak/ with versioning
            self.backup_manager.archive_run(current_run)

    def _try_submit_next_job(self):
        """Submit next job if ready"""
        if self.state.status == "running":
            return

        version = self.state.current_run

        # Prepare geometry BEFORE submission
        if not self.backup_manager.prepare_geometry_for_next_run(version):
            LOGGER.error(f"Failed to prepare geometry for run {version}")
            self.state.status = "failed"
            self.state.stats.failed_runs += 1
            return

        job_id = self.slurm.submit_job(self.calc_dir, self.state.config)

        if not job_id:
            LOGGER.error(f"Failed to submit job for run {version}")
            self.state.status = "failed"
            self.state.stats.failed_runs += 1
            return

        self.state.runs[version] = MDRunMetadata(
            run_number=version,
            job_id=job_id,
            submitted_at=datetime.now().isoformat(),
            status="running",
        )
        self.state.current_run += 1
        self.state.status = "running"

        LOGGER.info(f"✓ Submitted run {version} (job {job_id})")

    def _write_stopcar(self):
        """Write STOPCAR to gracefully stop VASP"""
        try:
            stopcar = self.calc_dir / "STOPCAR"
            stopcar.write_text("LABORT = .TRUE.\n")
            LOGGER.info(f"✓ Written STOPCAR to {self.calc_dir}")
        except Exception as e:
            LOGGER.error(f"Failed to write STOPCAR: {e}")

    def _print_summary(self):
        """Print summary of MD runs"""
        LOGGER.info("\nMD Run Summary:")
        LOGGER.info(f"  Total runs: {len(self.state.runs)}")
        LOGGER.info(f"  ✓ Completed: {self.state.stats.completed_runs}")
        LOGGER.info(f"  ✗ Failed: {self.state.stats.failed_runs}")

        if self.state.stats.run_durations:
            LOGGER.info(f"\nRun durations:")
            for run_num in sorted(self.state.stats.run_durations.keys()):
                duration = self.state.stats.run_durations[run_num]
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                LOGGER.info(f"  Run {run_num}: {hours:02d}:{minutes:02d}:{seconds:02d}")
