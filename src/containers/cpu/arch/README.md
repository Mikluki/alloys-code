# VASP Native Installation Instructions

## Block 1: System Dependencies

```bash
apt-get update && apt-get install -y \
    build-essential \
    csh \
    tcsh \
    curl \
    wget \
    tar \
    rsync \
    libfftw3-dev \
    zlib1g-dev \
    gawk \
    vim \
    git \
    cmake \
    unzip \
    libfabric1 \
    libnuma-dev \
    man \
    file \
    lsb-release \
    ca-certificates \
    gpg \
    libxcb-dri3-0 \
    libxcb1

apt-get clean && rm -rf /var/lib/apt/lists/*
```

## Block 2: Intel oneAPI Base Toolkit Installation

```bash
# Place l_BaseKit_p_2023.2.0.49397_offline.sh in /opt/
chmod +x /opt/l_BaseKit_p_2023.2.0.49397_offline.sh
/opt/l_BaseKit_p_2023.2.0.49397_offline.sh -a --silent --eula accept --install-dir /opt/intel/oneapi

# Verify MKL installation
if [ -f "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so" ]; then
    echo "✓ Intel MKL installed successfully"
else
    echo "✗ Intel MKL installation failed"
    exit 1
fi

# Cleanup
rm /opt/l_BaseKit_p_2023.2.0.49397_offline.sh
```

## Block 3: Intel oneAPI HPC Toolkit Installation

```bash
# Place l_HPCKit_p_2023.2.0.49440_offline.sh in /opt/
chmod +x /opt/l_HPCKit_p_2023.2.0.49440_offline.sh
/opt/l_HPCKit_p_2023.2.0.49440_offline.sh -a --silent --eula accept --install-dir /opt/intel/oneapi

# Verify classic compilers and MPI
for tool in ifort icc icpc mpiifort mpiicc mpiicpc; do
    if [ -f "/opt/intel/oneapi/compiler/latest/linux/bin/intel64/$tool" ] || [ -f "/opt/intel/oneapi/mpi/latest/bin/$tool" ]; then
        echo "✓ $tool found"
    else
        echo "✗ $tool not found"
    fi
done

# Cleanup
rm /opt/l_HPCKit_p_2023.2.0.49440_offline.sh
```

## Block 4: Environment Setup

```bash
# Add to /etc/environment or user profiles
cat >> /etc/environment << 'EOF'
ONEAPI_ROOT=/opt/intel/oneapi
MKLROOT=/opt/intel/oneapi/mkl/latest
INTEL_COMPILER_ROOT=/opt/intel/oneapi/compiler/latest
I_MPI_ROOT=/opt/intel/oneapi/mpi/latest
HDF5_ROOT=/opt/hdf5
FC=mpiifort
F90=mpiifort
F77=mpiifort
CC=mpiicc
CXX=mpiicpc
I_MPI_PIN=1
I_MPI_PIN_DOMAIN=auto
I_MPI_FABRICS=shm:ofi
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
KMP_AFFINITY=compact,1,0
EOF

# Add to /etc/bash.bashrc or user profiles
cat >> /etc/bash.bashrc << 'EOF'
export PATH=/opt/intel/oneapi/compiler/latest/linux/bin/intel64:/opt/intel/oneapi/compiler/latest/bin:/opt/intel/oneapi/mpi/latest/bin:/opt/hdf5/bin:/opt/vasp/vasp.6.4.3/bin:$PATH
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin:/opt/intel/oneapi/compiler/latest/linux/lib:/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/mpi/latest/lib:/opt/hdf5/lib:$LD_LIBRARY_PATH
EOF
```

## Block 5: HDF5 Build

```bash
# Source environment
source /etc/bash.bashrc

cd /tmp
wget https://support.hdfgroup.org/archive/support/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.22/src/hdf5-1.8.22.tar.gz
tar -xzf hdf5-1.8.22.tar.gz
cd hdf5-1.8.22

./configure \
    --prefix=/opt/hdf5 \
    --enable-fortran \
    --enable-shared \
    CC=icc FC=ifort CXX=icpc

make -j$(nproc)
make install

# Verify installation
if [ -f "/opt/hdf5/lib/libhdf5_fortran.so" ]; then
    echo "✓ HDF5 built and installed successfully"
else
    echo "✗ HDF5 installation failed"
fi

# Cleanup
cd / && rm -rf /tmp/hdf5-*
```

## Block 6: VASP Source Setup

```bash
# Place vasp.6.4.3.tgz and makefile.include.cpu.ifort in /opt/
mkdir -p /opt/vasp
cd /opt/vasp
tar -xzf /opt/vasp.6.4.3.tgz

# Copy makefile
cp /opt/makefile.include.cpu.ifort /opt/vasp/vasp.6.4.3/makefile.include

# Verify setup
if [ -d "/opt/vasp/vasp.6.4.3" ] && [ -f "/opt/vasp/vasp.6.4.3/makefile.include" ]; then
    echo "✓ VASP source extracted and makefile copied"
else
    echo "✗ VASP setup failed"
fi

# Cleanup
rm /opt/vasp.6.4.3.tgz /opt/makefile.include.cpu.ifort
```

## Compilation Note

After installation, VASP can be compiled with:

```bash
cd /opt/vasp/vasp.6.4.3
make DEPS=1 -j$(nproc)
```
