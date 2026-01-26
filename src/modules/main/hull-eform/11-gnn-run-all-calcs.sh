#!/bin/bash

# ============================================================================
# Orchestrator: Run GNN EOS relaxation for all calculators
# ============================================================================

set -e  # Exit on first error

PYTHON_SCRIPT="10-gnn-relax.py"
CALCULATORS=("mace" "orbv3" "sevennet" "esen" "nequip" "allegro" "dpa31" "grace2loaml")

echo "=========================================================================="
echo "GNN EOS Relaxation Multi-Calculator Orchestrator"
echo "=========================================================================="
echo ""

for calc in "${CALCULATORS[@]}"; do
    echo "[Calculator: $calc]"
    
    # Get venv path
    echo "  Getting venv path..."
    venv_path=$(python "$PYTHON_SCRIPT" --get-venv "$calc")
    
    if [ -z "$venv_path" ]; then
        echo "  ✗ Failed to get venv path for $calc"
        exit 1
    fi
    
    echo "  Venv: $venv_path"
    
    # Activate venv
    echo "  Activating venv..."
    source "$venv_path/bin/activate"
    
    # Run relaxation
    echo "  Running relaxation..."
    python "$PYTHON_SCRIPT" --calculator "$calc"
    exit_code=$?
    
    # Deactivate venv
    deactivate
    
    if [ $exit_code -ne 0 ]; then
        echo "  ✗ Failed with exit code $exit_code"
        exit $exit_code
    fi
    
    echo "  ✓ Complete"
    echo ""
done

echo "=========================================================================="
echo "All calculators complete!"
echo "=========================================================================="
