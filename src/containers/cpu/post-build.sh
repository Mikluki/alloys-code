#!/bin/bash
set -e  # Exit on any error

echo "Starting sandbox build..."
sudo singularity build --sandbox vasp.dev-cpu-ifort-hdf5 def/vasp.dev-cpu-ifort-hdf5.def

echo "Entering container to build VASP..."
sudo singularity exec --writable vasp.dev-cpu-ifort-hdf5 bash -c "cd /opt/vasp/vasp.6.4.3 && make DEPS=1 -j\$(nproc)"

echo "Building final SIF file..."
sudo singularity build vasp-cpu-ifort-hdf5.sif vasp.dev-cpu-ifort-hdf5

echo "Build sequence complete!"
