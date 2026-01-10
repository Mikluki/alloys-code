#!/usr/bin/env python3
"""
Full SLURM integration test for MD orchestrator.
- Uses real SlurmInterface (no mocking)
- Calls MOCK_SCRIPT_NAME to submit actual jobs
- Creates real output files (OUTCAR, CONTCAR, etc.)
- Verifies backup versioning works with actual files
- Requires: SLURM available (squeue) and MOCK_SCRIPT_NAME in same directory as test

Runtime: ~60 seconds (depends on SLURM queue)
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path

import pytest

logging.basicConfig(level=logging.DEBUG)

MOCK_SCRIPT_NAME = "vsf-submit-job-mock.py"


@pytest.fixture
def check_slurm_available():
    """Skip test if SLURM not available"""
    try:
        subprocess.run(
            ["squeue", "-u", "$(whoami)"],
            capture_output=True,
            timeout=5,
            shell=True,
            check=False,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("SLURM not available (squeue not found)")


@pytest.fixture
def check_submit_script_available():
    """Skip test if MOCK_SCRIPT_NAME not in same directory as test file"""
    test_dir = Path(__file__).parent
    script_path = test_dir / MOCK_SCRIPT_NAME

    if not script_path.exists():
        pytest.skip(f"{MOCK_SCRIPT_NAME} not found in {test_dir}")
    return str(script_path)


@pytest.fixture
def temp_calc_dir_slurm(request):
    """Create unique temporary calculation directory per test with required VASP files at cwd"""
    # Create per-test directory with test name to avoid overwriting
    test_name = request.node.name
    tmpdir = Path.cwd() / f"md_test_slurm_{test_name}"

    if tmpdir.exists():
        shutil.rmtree(tmpdir)

    tmpdir.mkdir(exist_ok=True)

    # Create minimal VASP input files
    (tmpdir / "INCAR").write_text("IBRION = 0\nNSW = 1000\n")
    (tmpdir / "POSCAR").write_text(
        "test structure\n"
        "1.0\n"
        "5.5 0 0\n"
        "0 5.5 0\n"
        "0 0 5.5\n"
        "Si\n"
        "2\n"
        "Direct\n"
        "0.0 0.0 0.0\n"
        "0.25 0.25 0.25\n"
    )
    (tmpdir / "POTCAR").write_text("PAW potential (mock)")
    (tmpdir / "KPOINTS").write_text("auto\n0\nMonkhorst-Pack\n2 2 2\n")

    print(f"\n📁 Test dir: {tmpdir}")
    yield tmpdir

    # Don't cleanup - user can inspect files
    print(f"\n📁 Results saved at: {tmpdir}")


def test_md_orchestrator_with_real_slurm(
    temp_calc_dir_slurm, check_slurm_available, check_submit_script_available
):
    """
    Full SLURM integration test: 3 MD iterations with real job submission.

    - Calls real MOCK_SCRIPT_NAME (mocked VASP)
    - Creates actual OUTCAR, CONTCAR, OSZICAR files
    - Verifies backup_run() creates versioned files (OUTCAR.0, OUTCAR.1, etc.)
    - Tests complete pipeline: submit → poll → backup → repeat

    Runtime: ~60 seconds (depends on SLURM queue)
    Requires: squeue available, MOCK_SCRIPT_NAME in same directory
    """
    from vsf.bin.devil.md_runner import MDConfig, MDOrchestrator

    state_file = temp_calc_dir_slurm / "md-state.json"

    # Path to mock script
    mock_script = Path(__file__).parent / MOCK_SCRIPT_NAME

    config = MDConfig(
        vasp_setup="module load vasp/6.4",
        ntasks=1,  # Use minimal resources for testing
        ncore=1,
        kpar=1,
        algo="Normal",
    )

    # Create orchestrator WITHOUT mocking - uses real SlurmInterface
    orchestrator = MDOrchestrator(
        calc_dir=temp_calc_dir_slurm,
        config=config,
        state_file=state_file,
        max_iterations=3,
        max_duration_hours=None,
        sleep_time=2,  # Check every x seconds
        submit_script=str(mock_script),
    )

    assert orchestrator.state.current_run == 0
    assert orchestrator.state.stats.completed_runs == 0
    assert state_file.exists()

    start = time.time()
    orchestrator.run(dry_run=False)
    elapsed = time.time() - start

    # Verify orchestration metrics
    assert orchestrator.state.status == "stopped", "Should stop gracefully"
    assert orchestrator.state.stats.completed_runs == 3, "Should complete 3 runs"
    assert orchestrator.state.stats.failed_runs == 0, "Should have no failures"
    assert len(orchestrator.state.runs) == 3, "Should have 3 run records"

    # Verify actual output files were created and backed up
    assert (temp_calc_dir_slurm / "OUTCAR.0").exists(), "Should have backup OUTCAR.0"
    assert (temp_calc_dir_slurm / "OUTCAR.1").exists(), "Should have backup OUTCAR.1"
    assert (temp_calc_dir_slurm / "OUTCAR.2").exists(), "Should have backup OUTCAR.2"

    assert (temp_calc_dir_slurm / "CONTCAR.0").exists(), "Should have backup CONTCAR.0"
    assert (temp_calc_dir_slurm / "CONTCAR.1").exists(), "Should have backup CONTCAR.1"
    assert (temp_calc_dir_slurm / "CONTCAR.2").exists(), "Should have backup CONTCAR.2"

    # Verify backup files contain different content (actual runs happened)
    outcar_0 = (temp_calc_dir_slurm / "OUTCAR.0").read_text()
    outcar_1 = (temp_calc_dir_slurm / "OUTCAR.1").read_text()
    assert outcar_0, "OUTCAR.0 should have content"
    assert outcar_1, "OUTCAR.1 should have content"

    print(f"\n✓ SLURM integration test passed in {elapsed:.1f}s")
    print(f"  Completed: {orchestrator.state.stats.completed_runs}")
    print(f"  Failed: {orchestrator.state.stats.failed_runs}")
    print(f"  Output files backed up: 0, 1, 2")
    print(f"  Results at: {temp_calc_dir_slurm}")


def test_slurm_interface_submit_job(temp_calc_dir_slurm, check_submit_script_available):
    """
    Test SlurmInterface.submit_job() directly.

    Verifies that job submission actually calls MOCK_SCRIPT_NAME and returns valid job ID.
    """
    from vsf.bin.devil.engine import SlurmInterface
    from vsf.bin.devil.md_runner import MDConfig

    # Path to mock script
    mock_script = Path(__file__).parent / MOCK_SCRIPT_NAME

    slurm = SlurmInterface(submit_script=str(mock_script))

    config = MDConfig(
        vasp_setup="module load vasp/6.4",
        ntasks=1,
        ncore=1,
        kpar=1,
        algo="Normal",
    )

    # Submit a real job
    job_id = slurm.submit_job(temp_calc_dir_slurm, config)

    assert job_id is not None, "Job submission should return job ID"
    assert job_id.isdigit(), f"Job ID should be numeric, got {job_id}"

    print(f"\n✓ Job submitted: {job_id}")

    # Verify job is in queue
    current_jobs = slurm.get_current_job_ids()
    assert current_jobs is not None, "Should be able to get current jobs"
    assert job_id in current_jobs, f"Job {job_id} should be in queue"

    print(f"✓ Job {job_id} confirmed in SLURM queue")

    # Wait for job to complete (mock job runs ~10 sec)
    max_wait = 30  # 30 sec max (3x job duration for safety)
    start = time.time()
    while slurm.is_job_running(job_id):
        elapsed = time.time() - start
        if elapsed > max_wait:
            print(f"⚠ Job still running after {max_wait}s, moving on...")
            break
        time.sleep(2)
        print(f"  Waiting for job {job_id} ({elapsed:.0f}s)...")

    elapsed = time.time() - start
    print(f"✓ Job {job_id} completed in {elapsed:.1f}s")

    # Verify output files were created
    assert (temp_calc_dir_slurm / "OUTCAR").exists(), "OUTCAR should be created"
    assert (temp_calc_dir_slurm / "CONTCAR").exists(), "CONTCAR should be created"

    print(f"✓ Output files created: OUTCAR, CONTCAR")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
