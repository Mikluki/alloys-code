"""
core.py
These define the contract. They rarely change.
Everything else depends on them.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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


@dataclass
class StageState:
    """State of a single stage execution"""

    name: str
    status: str  # "pending", "running", "completed", "failed"
    target_dir: Optional[Path] = None
    job_id: Optional[str] = None
    executed_at: Optional[str] = None  # ISO format
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "target_dir": str(self.target_dir) if self.target_dir else None,
            "job_id": self.job_id,
            "executed_at": self.executed_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StageState":
        return cls(
            name=data["name"],
            status=data["status"],
            target_dir=Path(data["target_dir"]) if data.get("target_dir") else None,
            job_id=data.get("job_id"),
            executed_at=data.get("executed_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )


@dataclass
class WorkflowInstance:
    """State of one workflow run (one directory)"""

    workflow_name: str
    status: str  # "pending", "in_progress", "completed", "failed"
    current_stage: int
    stages: List[StageState]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "status": self.status,
            "current_stage": self.current_stage,
            "stages": [s.to_dict() for s in self.stages],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "WorkflowInstance":
        return cls(
            workflow_name=data["workflow_name"],
            status=data["status"],
            current_stage=data["current_stage"],
            stages=[StageState.from_dict(s) for s in data["stages"]],
        )


@dataclass
class WorkflowState:
    """Complete state for all workflows in a run"""

    config: Dict[str, Any]
    workflow_name: str
    created_at: str  # ISO format
    workflow_instances: Dict[str, WorkflowInstance]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "workflow_name": self.workflow_name,
            "created_at": self.created_at,
            "workflow_instances": {
                k: v.to_dict() for k, v in self.workflow_instances.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "WorkflowState":
        return cls(
            config=data["config"],
            workflow_name=data["workflow_name"],
            created_at=data["created_at"],
            workflow_instances={
                k: WorkflowInstance.from_dict(v)
                for k, v in data["workflow_instances"].items()
            },
        )


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
# Workflow Definitions
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


@dataclass
class LoopingWorkflow(Workflow):
    """A workflow that repeats until a termination condition is met"""

    max_iterations: Optional[int] = None
    max_duration_hours: Optional[float] = None

    def validate(self):
        """Validate looping workflow structure"""
        super().validate()

        if not self.max_iterations and not self.max_duration_hours:
            raise ValueError(
                "LoopingWorkflow must have max_iterations or max_duration_hours"
            )


# ============================================================================
# CLI utilities and Main
# ============================================================================


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
        LOGGER.error(f"Error processing glob pattern '{glob_pattern}': {e}")
        return []

    return directories
