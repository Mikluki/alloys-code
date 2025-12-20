#!/bin/bash

# Script to generate KPOINTS files for VASP calculations
# Usage: ./generate_kpoints.sh <target_directory>
# 
# Finds directories ending with -XXX (3 digits) and creates KPOINTS files
# with Gamma-centered k-point grids based on the suffix digits

# Check if target directory is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <target_directory>"
    echo "Example: $0 /path/to/calculations"
    exit 1
fi

target_dir="$1"

# Check if target directory exists
if [ ! -d "$target_dir" ]; then
    echo "Error: Directory '$target_dir' does not exist"
    exit 1
fi

echo "Searching for directories with -XXX suffix in: $target_dir"
echo "Generating Gamma-centered KPOINTS files..."

# Counter for processed directories
count=0

# Find directories ending with -[0-9][0-9][0-9]
find "$target_dir" -maxdepth 1 -type d -name "*-[0-9][0-9][0-9]" | while read -r dir; do
    # Extract directory name
    dirname=$(basename "$dir")
    
    # Extract the 3-digit suffix after the last dash
    suffix="${dirname##*-}"
    
    # Check if suffix is exactly 3 digits
    if [[ ! "$suffix" =~ ^[0-9]{3}$ ]]; then
        echo "Warning: Skipping '$dirname' - suffix '$suffix' is not exactly 3 digits"
        continue
    fi
    
    # Split the 3 digits into kx, ky, kz
    kx="${suffix:0:1}"
    ky="${suffix:1:1}"
    kz="${suffix:2:1}"
    
    # Create KPOINTS file path
    kpoints_file="$dir/KPOINTS"
    
    # Generate KPOINTS file content
    cat > "$kpoints_file" << EOF
Automatic mesh
0
Gamma
$kx $ky $kz
0.0 0.0 0.0
EOF
    
    echo "Created KPOINTS in $dirname: k-grid = $kx $ky $kz"
    ((count++))
done

echo "Processed $count directories"
