#!/bin/bash
# Quick start: Run MD orchestrator tests
# Usage: ./run_tests.sh

set -e

echo "=========================================="
echo "MD Orchestrator Test Suite"
echo "=========================================="
echo

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python: $python_version"

# Check pytest
if ! command -v pytest &> /dev/null; then
    echo "✗ pytest not found. Installing..."
    uv pip install pytest pytest-mock
fi
echo "✓ pytest installed"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo
echo "=========================================="
echo "Test 1: State Persistence (< 1 sec)"
echo "=========================================="
pytest test_md_setup.py::test_state_persistence -v -s

echo
echo "=========================================="
echo "Test 2: Backup Manager (< 1 sec)"
echo "=========================================="
pytest test_md_setup.py::test_backup_manager -v -s

echo
echo "=========================================="
echo "Test 3: Full Orchestrator (~ 45 sec)"
echo "=========================================="
pytest test_md_setup.py::test_md_orchestrator_3_iterations -v -s

echo
echo "=========================================="
echo "Summary"
echo "=========================================="
pytest test_md_setup.py -v --tb=short

echo
echo "✓ All tests passed!"
