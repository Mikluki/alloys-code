# VASP Devil Prototype - Architecture Test

This prototype demonstrates the complete workflow architecture for managing VASP calculations with proper separation of concerns.

## What This Prototype Proves

✅ **Clean data flow** - No hidden state, explicit contracts  
✅ **Composable workflows** - Build complex flows from simple stages  
✅ **Independent progression** - Each directory advances through stages independently  
✅ **Resilient execution** - Handles missing files gracefully  
✅ **Extensible design** - Adding new stage types is straightforward  

## Architecture Overview

```
User defines workflow → Devil orchestrates → Stages execute
                            ↓
                    State persisted to JSON
                            ↓
                    Resume from any point
```

### Core Components

1. **Stage** - A single step (submit job, copy files, validate)
2. **Workflow** - Ordered sequence of stages
3. **Context** - Immutable data passed to stages
4. **Result** - What stages return (success, target_dir, job_id)
5. **Engine** - Executes stages and manages state transitions
6. **Orchestrator** - Main loop that coordinates everything

## Files

- `vasp_devil_prototype.py` - Complete implementation (850 lines)
- `test_prototype.py` - Unit tests (no SLURM required)
- `PROTOTYPE_README.md` - This file

## Quick Start

### 1. Run Unit Tests

```bash
python test_prototype.py
```

This tests all components without requiring SLURM or VASP.

### 2. Create Demo Setup

The test script can create mock VASP directories:

```bash
python test_prototype.py
# Answer 'y' when prompted to create demo setup
```

### 3. Test with Mock Submission Script

Create a mock `vsf-submit-job.py` for testing:

```bash
cat > vsf-submit-job.py << 'EOF'
#!/usr/bin/env python3
import sys
import random
# Mock submission - just print what would be done
print(f"✓ Submitted job 'mock_job' with ID {random.randint(10000, 99999)} for {sys.argv[1]}")
EOF
chmod +x vsf-submit-job.py
```

Then run dry-run:

```bash
python vasp_devil_prototype.py 'demo_vasp_calcs/struct_*' \
  16 4 2 Normal \
  --vasp-setup 'module load vasp' \
  --workflow double \
  --max-jobs 2 \
  --dry-run
```

## Usage Examples

### Simple Workflow (Single Relaxation)

```bash
python vasp_devil_prototype.py 'calcs/rand*' \
  16 4 2 Normal \
  --vasp-setup "module load vasp/6.4.3" \
  --workflow simple \
  --max-jobs 10 \
  --partition cpu
```

**Stages:**
1. Submit relaxation job
2. Validate completion

### Double Relaxation Workflow

```bash
python vasp_devil_prototype.py 'calcs/struct_*' \
  16 4 2 Normal \
  --vasp-setup "module load vasp/6.4.3" \
  --workflow double \
  --max-jobs 5 \
  --partition cpu
```

**Stages:**
1. Submit initial relaxation (`struct_A/`)
2. Validate completion
3. Copy files to `struct_A_relax2/` (CONTCAR→POSCAR, WAVECAR, etc.)
4. Submit final relaxation (`struct_A_relax2/`)
5. Validate completion

### Resume Interrupted Run

The state is automatically saved. Just run the same command again:

```bash
# Interrupted run - state saved to vasp-devil-state.json
^C

# Resume - picks up where it left off
python vasp_devil_prototype.py 'calcs/struct_*' \
  16 4 2 Normal \
  --vasp-setup "module load vasp/6.4.3" \
  --workflow double \
  --max-jobs 5
```

## State File Structure

The `vasp-devil-state.json` contains everything:

```json
{
  "workflow_name": "double_relaxation",
  "workflow_instances": {
    "/path/to/struct_A": {
      "workflow_name": "double_relaxation",
      "status": "in_progress",
      "current_stage": 2,
      "stages": [
        {
          "name": "relax1",
          "status": "completed",
          "job_id": "12345",
          "target_dir": "/path/to/struct_A",
          "executed_at": "2025-10-06T10:00:00"
        },
        {
          "name": "validate1",
          "status": "completed",
          "target_dir": "/path/to/struct_A"
        },
        {
          "name": "copy_for_relax2",
          "status": "running",
          "target_dir": "/path/to/struct_A_relax2"
        },
        ...
      ]
    },
    "/path/to/struct_B": {
      ...
    }
  }
}
```

## Key Design Decisions

### 1. Immutable Context

```python
@dataclass(frozen=True)
class StageContext:
    source_dir: Path
    workflow_name: str
    stage_index: int
    previous_target_dir: Optional[Path] = None
    previous_job_id: Optional[str] = None
    global_config: Dict[str, Any]
```

**Why frozen?** Stages can't accidentally modify context and cause hidden coupling.

### 2. Explicit Results

```python
@dataclass
class StageResult:
    success: bool
    target_dir: Path
    job_id: Optional[str] = None
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
```

**Why explicit?** The only way stages communicate. No side channels, no magic.

### 3. No Metadata Propagation

Metadata is logged but NOT passed to next stage. Stages discover what they need from the filesystem.

**Why?** Prevents metadata from becoming a dumping ground that grows unbounded.

### 4. Directory Resolution

```python
DirectoryResolver.resolve("{source_dir}_relax2", context)
# → /calc/struct_A_relax2
```

**Why centralized?** Consistent naming conventions, easy to change later.

## Adding New Stage Types

### Example: ScriptStage (Run Custom Script)

```python
@dataclass
class ScriptStageConfig:
    script_path: str
    args_template: str  # e.g., "{prev_dir} {source_dir}"
    target_dir_expr: str

class ScriptStage(Stage):
    def __init__(self, name: str, config: ScriptStageConfig):
        super().__init__(name)
        self.config = config
    
    def execute(self, context: StageContext) -> StageResult:
        target_dir = DirectoryResolver.resolve(
            self.config.target_dir_expr, context
        )
        
        # Build command with resolved args
        args = self.config.args_template.format(
            prev_dir=context.previous_target_dir,
            source_dir=context.source_dir
        )
        
        cmd = [self.config.script_path] + args.split()
        
        try:
            subprocess.run(cmd, check=True, cwd=target_dir)
            return StageResult(success=True, target_dir=target_dir)
        except Exception as e:
            return StageResult(
                success=False,
                target_dir=target_dir,
                error_message=str(e)
            )
    
    def requires_job_completion(self) -> bool:
        return False  # Script runs instantly
```

**Usage:**

```python
workflow = Workflow(
    name="custom",
    stages=[
        SubmitStage("relaxation", ...),
        ValidationStage("validate"),
        ScriptStage(
            "generate_kpoints",
            ScriptStageConfig(
                script_path="./generate_kpoints.py",
                args_template="{prev_dir}",
                target_dir_expr="{source_dir}_static"
            )
        ),
        SubmitStage("static_calc", ...)
    ]
)
```

## Testing Strategy

### Unit Tests (No SLURM)

```bash
python test_prototype.py
```

Tests individual components in isolation.

### Integration Test (With Mock SLURM)

1. Create mock `vsf-submit-job.py`
2. Create mock `squeue` command
3. Run prototype with mock directories

### Full Test (On Cluster)

1. Create small test structures
2. Run with `--max-jobs 1` first
3. Monitor with actual SLURM commands

## What's Different from Original?

| Aspect | Original | Prototype |
|--------|----------|-----------|
| Architecture | God object | Separated concerns |
| Workflows | Hardcoded single | Composable stages |
| State | Flat directory list | Per-workflow instances |
| Stage progression | All lock-step | Independent per-dir |
| Error handling | Placeholders | Explicit results |
| Testing | Hard | Components testable |
| Extension | Modify core | Add new stages |

## Limitations & Future Work

### Current Limitations

1. **No conditional branching** - Stages run in fixed order
2. **No parallel stages** - Could run multiple calcs in parallel within one workflow
3. **No retry logic** - Failed stage = failed workflow
4. **No stage dependencies** - Can't say "stage 3 needs output from stage 1"

### Future Extensions

1. **Conditional stages**
   ```python
   class ConditionalStage(Stage):
       def can_execute(self, context):
           # Check if OUTCAR shows convergence issues
           return check_convergence(context.previous_target_dir)
   ```

2. **Parallel execution**
   ```python
   class ParallelStages(Stage):
       """Run multiple stages concurrently"""
       def __init__(self, name: str, stages: List[Stage]):
           ...
   ```

3. **Retry with backoff**
   ```python
   @dataclass
   class RetryConfig:
       max_attempts: int
       backoff_seconds: float
   ```

4. **Stage groups/sub-workflows**
   ```python
   def create_convergence_workflow():
       return Workflow(
           stages=[
               SubmitStage(..., encut=400),
               ValidationStage(),
               SubmitStage(..., encut=500),
               ValidationStage(),
               CompareResultsStage()
           ]
       )
   ```

## Questions & Answers

### Q: Why not use an existing workflow engine?

**A:** Domain-specific needs (VASP file conventions, SLURM integration) and learning architecture through implementation.

### Q: What if I need to pass complex data between stages?

**A:** Write it to the filesystem. Stages discover what they need by reading files. This keeps coupling minimal.

### Q: Can stages run in parallel?

**A:** Not yet, but the architecture supports it. Each workflow instance already progresses independently.

### Q: How do I debug a failed workflow?

**A:** 
1. Check the log file
2. Inspect `vasp-devil-state.json` for the failed stage
3. Look at the stage's `error` field
4. Check files in the `target_dir`

### Q: Can I modify a workflow while it's running?

**A:** No. The workflow definition is captured at start. Stop the devil, modify code, restart.

## Next Steps

1. **Test the prototype** with your real VASP setup
2. **Identify pain points** - What's missing? What's awkward?
3. **Implement additional workflows** - Convergence tests? Band structures?
4. **Add stage types** - What operations do you need beyond submit/copy/validate?
5. **Refine error handling** - What failures need special treatment?

## Success Criteria

The architecture is successful if:

✅ Adding a new workflow type takes < 10 lines of code  
✅ Adding a new stage type is self-contained (no core changes)  
✅ Debugging failed workflows is straightforward  
✅ State file is human-readable and inspectable  
✅ You feel confident modifying and extending the code  

---

**Ready to test?** Run `python test_prototype.py` and see the architecture in action!
