#!/usr/bin/env python3
"""
Test script for VASP Devil prototype
Creates mock VASP directories and tests the workflow without SLURM
"""

import tempfile
from pathlib import Path


# Mock directories for testing
def create_mock_vasp_directory(base_dir: Path, name: str):
    """Create a mock VASP calculation directory"""
    calc_dir = base_dir / name
    calc_dir.mkdir(exist_ok=True)

    # Create mock input files
    (calc_dir / "INCAR").write_text("NCORE = 4\nNSW = 100\n")
    (calc_dir / "POSCAR").write_text("Mock POSCAR\n")
    (calc_dir / "POTCAR").write_text("Mock POTCAR\n")
    (calc_dir / "KPOINTS").write_text("Mock KPOINTS\n")

    print(f"✓ Created mock directory: {calc_dir}")
    return calc_dir


def create_mock_outcar(calc_dir: Path, elapsed_time: float = 3600.0):
    """Create a mock OUTCAR with completion marker"""
    outcar_content = f"""
 vasp.6.4.3
 
 ... (mock VASP output) ...
 
 Total CPU time used (sec):      {elapsed_time}
 User time (sec):                {elapsed_time * 0.9}
 System time (sec):              {elapsed_time * 0.1}
 Elapsed time (sec):             {elapsed_time}
"""
    (calc_dir / "OUTCAR").write_text(outcar_content)
    print(f"✓ Created mock OUTCAR in {calc_dir}")


def create_mock_contcar(calc_dir: Path):
    """Create a mock CONTCAR"""
    (calc_dir / "CONTCAR").write_text("Mock relaxed structure\n")
    print(f"✓ Created mock CONTCAR in {calc_dir}")


def test_directory_resolution():
    """Test DirectoryResolver"""
    from core import DirectoryResolver, StageContext

    print("\n=== Testing DirectoryResolver ===")

    context = StageContext(
        source_dir=Path("/calc/struct_A"),
        workflow_name="test",
        stage_index=0,
        previous_target_dir=Path("/calc/struct_A"),
    )

    tests = [
        ("{source_dir}", Path("/calc/struct_A")),
        ("{source_dir}_relax2", Path("/calc/struct_A_relax2")),
        ("{prev_dir}/subdir", Path("/calc/struct_A/subdir")),
    ]

    for expr, expected in tests:
        result = DirectoryResolver.resolve(expr, context)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{expr}' → {result}")


def test_copy_stage():
    """Test CopyStage functionality"""
    from vsf.bin.devil.core import StageContext
    from vsf.bin.devil.core_stages import CopyStage

    print("\n=== Testing CopyStage ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create source directory with files
        source_dir = tmpdir / "source"
        source_dir.mkdir()
        (source_dir / "CONTCAR").write_text("structure")
        (source_dir / "INCAR").write_text("params")
        (source_dir / "WAVECAR").write_text("waves")

        # Create stage
        stage = CopyStage.for_vasp_continuation(
            "test_copy", target_dir_expr="{source_dir}_copy"
        )

        # Create context
        context = StageContext(
            source_dir=tmpdir / "source",
            workflow_name="test",
            stage_index=1,
            previous_target_dir=source_dir,
        )

        # Execute
        result = stage.execute(context)

        if result.success:
            print(f"✓ Copy succeeded: {result.metadata['copied']}")
            print(f"  Target: {result.target_dir}")

            # Check files
            target = result.target_dir
            if (target / "POSCAR").exists():
                print("  ✓ CONTCAR renamed to POSCAR")
            if (target / "INCAR").exists():
                print("  ✓ INCAR copied")
        else:
            print(f"✗ Copy failed: {result.error_message}")


def test_validation_stage():
    """Test ValidationStage"""
    from vsf.bin.devil.core import StageContext
    from vsf.bin.devil.core_stages import ValidationStage

    print("\n=== Testing ValidationStage ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test 1: Missing OUTCAR
        print("\nTest 1: Missing OUTCAR")
        calc_dir1 = tmpdir / "calc1"
        calc_dir1.mkdir()

        stage = ValidationStage("validate")
        context = StageContext(
            source_dir=calc_dir1,
            workflow_name="test",
            stage_index=1,
            previous_target_dir=calc_dir1,
        )

        result = stage.execute(context)
        print(
            f"  {'✓' if not result.success else '✗'} Correctly failed: {result.error_message}"
        )

        # Test 2: Incomplete OUTCAR
        print("\nTest 2: Incomplete OUTCAR")
        calc_dir2 = tmpdir / "calc2"
        calc_dir2.mkdir()
        (calc_dir2 / "OUTCAR").write_text("Some output but not complete")

        context2 = StageContext(
            source_dir=calc_dir2,
            workflow_name="test",
            stage_index=1,
            previous_target_dir=calc_dir2,
        )

        result2 = stage.execute(context2)
        print(f"  {'✓' if not result2.success else '✗'} Correctly detected incomplete")

        # Test 3: Complete OUTCAR
        print("\nTest 3: Complete OUTCAR")
        calc_dir3 = tmpdir / "calc3"
        calc_dir3.mkdir()
        create_mock_outcar(calc_dir3, elapsed_time=1234.5)

        context3 = StageContext(
            source_dir=calc_dir3,
            workflow_name="test",
            stage_index=1,
            previous_target_dir=calc_dir3,
        )

        result3 = stage.execute(context3)
        print(f"  {'✓' if result3.success else '✗'} Correctly validated")
        if result3.metadata:
            print(f"  Elapsed time: {result3.metadata.get('elapsed_time_sec')}s")


def test_workflow_creation():
    """Test workflow creation"""
    from stages import create_double_relaxation_workflow, create_simple_workflow

    print("\n=== Testing Workflow Creation ===")

    # Simple workflow
    simple = create_simple_workflow(16, 4, 2, "Normal")
    print(f"\n✓ Simple workflow: {simple.name}")
    print(f"  Description: {simple.description}")
    print(f"  Stages: {len(simple.stages)}")
    for i, stage in enumerate(simple.stages):
        job_marker = " [JOB]" if stage.requires_job_completion() else ""
        print(f"    {i+1}. {stage.name}{job_marker}")

    # Double relaxation workflow
    double = create_double_relaxation_workflow(16, 4, 2, "Normal")
    print(f"\n✓ Double relaxation workflow: {double.name}")
    print(f"  Description: {double.description}")
    print(f"  Stages: {len(double.stages)}")
    for i, stage in enumerate(double.stages):
        job_marker = " [JOB]" if stage.requires_job_completion() else ""
        print(f"    {i+1}. {stage.name}{job_marker}")


def test_state_management():
    """Test state creation and persistence"""
    from vsf.bin.devil.core_stages import create_simple_workflow
    from vsf.bin.devil.engine import StateManager

    print("\n=== Testing State Management ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        state_file = tmpdir / "test_state.json"
        manager = StateManager(state_file)

        # Create workflow and directories
        workflow = create_simple_workflow(16, 4, 2, "Normal")
        directories = [tmpdir / f"calc_{i}" for i in range(3)]
        for d in directories:
            d.mkdir()

        config = {
            "max_jobs": 10,
            "vasp_setup": "module load vasp",
            "ntasks": 16,
            "ncore": 4,
            "kpar": 2,
            "algo": "Normal",
        }

        # Create state
        state = manager.create_initial_state(config, workflow, directories)
        print(f"✓ Created state with {len(state.workflow_instances)} instances")

        # Save state
        success = manager.save(state)
        print(f"{'✓' if success else '✗'} Saved state to {state_file}")

        # Load state
        loaded = manager.load()
        print(f"{'✓' if loaded else '✗'} Loaded state from file")

        if loaded:
            print(f"  Workflow name: {loaded.workflow_name}")
            print(f"  Instances: {len(loaded.workflow_instances)}")

            # Show first instance
            first_dir = str(directories[0])
            first_inst = loaded.workflow_instances[first_dir]
            print(f"\n  First instance ({first_dir}):")
            print(f"    Status: {first_inst.status}")
            print(f"    Current stage: {first_inst.current_stage}")
            print(f"    Stages:")
            for stage in first_inst.stages:
                print(f"      - {stage.name}: {stage.status}")


def create_demo_setup():
    """Create a demo directory structure for manual testing"""
    print("\n=== Creating Demo Setup ===")

    demo_dir = Path("./demo_vasp_calcs")
    if demo_dir.exists():
        print(f"Demo directory already exists: {demo_dir}")
        return demo_dir

    demo_dir.mkdir()

    # Create 3 mock calculation directories
    for i in range(1, 4):
        calc_dir = create_mock_vasp_directory(demo_dir, f"struct_{i}")

    print(f"\n✓ Demo setup created in: {demo_dir}")
    print("\nTo test with the prototype, run:")
    print(f"  python run_double_relaxation.py")
    print(f"  (after editing DIRECTORIES_PATTERN to 'demo_vasp_calcs/struct_*')")

    return demo_dir


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("VASP Devil Prototype - Unit Tests")
    print("=" * 70)

    try:
        test_directory_resolution()
        test_copy_stage()
        test_validation_stage()
        test_workflow_creation()
        test_state_management()

        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)

        # Offer to create demo setup
        print("\nWould you like to create a demo setup? (y/n)")
        response = input().strip().lower()
        if response == "y":
            create_demo_setup()

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Make sure core.py, stages.py, and engine.py are in the same directory")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
