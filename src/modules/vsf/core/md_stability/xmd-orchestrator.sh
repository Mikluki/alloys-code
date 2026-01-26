#!/bin/bash

# Resolve YAML file: check cwd first, then home
if [ -f "energy_sources.yaml" ]; then
    YAML_FILE="energy_sources.yaml"
elif [ -f "$HOME/energy_sources.yaml" ]; then
    YAML_FILE="$HOME/energy_sources.yaml"
else
    echo "ERROR: No energy_sources.yaml found in cwd or home directory"
    exit 1
fi

echo "Using config: $YAML_FILE"
echo ""

# Helper: Extract venv path for a given energy source
get_venv_path() {
    local source=$1
    python3 << EOF
import yaml

with open("$YAML_FILE") as f:
    config = yaml.safe_load(f)
    path = config.get('energy_sources', {}).get('$source', {}).get('venv')
    if path:
        print(path.replace('~', '$HOME'))
    else:
        raise ValueError(f"Energy source '$source' not found in YAML")
EOF
}

# Helper: Get list of all energy sources from YAML
get_all_sources() {
    python3 << EOF
import yaml

with open("$YAML_FILE") as f:
    config = yaml.safe_load(f)
    sources = list(config.get('energy_sources', {}).keys())
    print(' '.join(sources))
EOF
}

# Determine which sources to run
if [ $# -eq 0 ]; then
    # No arguments: run all sources from YAML
    sources=$(get_all_sources 2>&1)
    if [ $? -ne 0 ]; then
        echo "ERROR getting sources from YAML:"
        echo "$sources"
        exit 1
    fi
else
    # Arguments provided: use those sources
    sources="$@"
fi

# Run main.py for each source
for source in $sources; do
    echo "=========================================="
    echo "Running: $source"
    echo "=========================================="
    
    venv_path=$(get_venv_path "$source")
    
    # Source the venv
    source "$venv_path/bin/activate"
    
    # Run main
    python main.py --energy-source "$source"
    
    # Deactivate venv
    deactivate
    
    echo ""
done
