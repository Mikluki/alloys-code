## Architecture Summary

**Core Concept:**
Workflow orchestration using transparent stages. Each stage is a discrete, readable operation (submit job, validate completion, backup files, etc.). Workflows are ordered sequences of stages executed by an orchestrator. State persists as JSON between runs.

**Two Workflow Types:**

1. **Batch Workflows** (static/relaxation jobs):
   - One-shot sequences: relax → validate → copy → relax → validate
   - Iterate over multiple structures
   - WorkflowEngine in engine.py manages job submission limits and polling

2. **Looping Workflows** (MD runs):
   - Repeat stages until condition met (max iterations OR max duration hours)
   - Single structure, same directory
   - MDOrchestrator (in md_runner.py) handles loop termination and time tracking

**Files & Responsibilities:**

- **core.py**: Abstract interfaces (Stage, Workflow, LoopingWorkflow), data structures (StageContext, StageResult), directory resolution. No implementations.

- **stages.py**: Batch workflow stages (SubmitStage, CopyStage, ValidationStage) + factories (create_simple_workflow, create_double_relaxation_workflow)

- **md_stages.py**: MD workflow stages (MDSubmitStage, MDValidationStage, MDBackupStage, MDPrepareStage) + factory (create_md_looping_workflow)

- **engine.py**: WorkflowEngine (executes batch stages), SlurmInterface (submit_job, is_job_running, get_current_job_ids), StateManager (JSON persistence with dataclass serialization), VASPDevil (main orchestrator for batch)

- **md_runner.py**: MDBackupManager (file versioning), MDState/MDConfig/MDStats (dataclasses), MDStateManager (persistence), MDOrchestrator (main loop for MD with timer/iteration checks)

- **run_md.py & run_double_relaxation.py**: Entry points. Configuration constants at top (CALC_DIR, MAX_DURATION_HOURS, etc.). Build config objects and instantiate orchestrator.

**Conventions:**

- `LOGGER = logging.getLogger(__name__)` at top of every file
- Dataclasses everywhere (config, state, metadata). Always implement `to_dict()`/`from_dict()` for JSON round-trip
- Configuration as module-level constants in run files (no CLI args, direct edit for iteration)
- StageContext passes previous stage outputs forward (job_id, target_dir, etc.)
- All persistence is atomic (write to .tmp, then rename)

**Key Design Principles:**

- Fail fast with clear error messages
- Explicit configuration over magic behavior
- Flat data structures (no deep nesting)

Main files
- core.py
- stages.py
- engine.py
- md_stages.py
- md_runner.py
- run_md.py
