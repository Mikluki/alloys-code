# VASP GPU Container Build Guide

## Quick setup

- **Step 1: Build sandbox first**

  ```bash
  sudo singularity build --sandbox vasp-dev-dir vasp-gpu.def
  ```

- **Step 2: Enter and test environment**

  ```bash
  sudo singularity shell --writable vasp-dev-dir

  # Test if environment is automatically set (should work now!)
  which mpif90
  echo $MKLROOT
  which vasp_std  # This should be empty since VASP isn't compiled yet
  ```

- **Step 3: Compile VASP**

  ```bash
  cd /opt/vasp/vasp.6.4.2
  make DEPS=1 -j$(nproc)
  ```

  > Compilation time: ~10-15 minutes depending on CPU cores

  > [!TIP] Expected behavior:
  >
  > - Dependency generation phase
  > - Fortran compilation of VASP modules
  > - Linking with MKL and CUDA libraries
  > - Creation of `vasp_std`, `vasp_gam`, `vasp_ncl` executables

- **Step 4: Check build**

  ```bash
  # Re-enter the sandbox container where you compiled VASP
  singularity shell --writable vasp-dev-dir
  ls -la /opt/vasp/vasp.6.4.2/bin/
  file /opt/vasp/vasp.6.4.2/bin/vasp_std  # checck if executable

  which vasp_std
  timeout 5s vasp_std 2>&1 | head -5
  ```

  This should show VASP starting up (and complaining about missing input files, which is normal).

  > [!NOTE]
  > If you need, edit the environment file directly:
  >
  > ```bash
  > sudo vim vasp-dev-dir/.singularity.d/env/90-environment.sh
  > ```

- **Step 5: Build final .sif**

  ```bash
  exit
  singularity build vasp-gpu.sif vasp-dev-dir
  ```

## Resources

### Prerequisites

**Required source files in `src/` directory:**

```
src/
├── nvhpc_2025_253_Linux_x86_64_cuda_12.8.tar.gz    # NVIDIA HPC SDK (7.9GB)
├── vasp.6.4.2.tgz                                   # VASP source code (94MB)
└── makefile.include                                 # VASP makefile template (pre-configured for MKL paths)
```

**Note:** The `makefile.include` template should be pre-configured with correct MKL paths:

```makefile
MKLROOT    ?= /opt/intel/oneapi/mkl/latest
LLIBS_MKL   = -Mmkl -L$(MKLROOT)/lib -lmkl_scalapack_lp64 -lmkl_blacs_openmpi_lp64
```

**System requirements:**

- Singularity/Apptainer installed
- Internet connection (for Intel MKL APT installation)
- Sufficient disk space (~15GB for build process)

## Troubleshooting

### Common Issues and Solutions

1. Permission denied errors: build & launch do with **sudo**

2. Intel setvars.sh script errors:

   - Skip automatic sourcing, set environment variables manually

3. NVIDIA warnings during container entry:

   - Add `--nv` flag only if running on NVIDIA GPU systems
   - For CPU-only builds, ignore the warnings

4. VASP binaries not found:

   - Add to PATH: `export PATH=/opt/vasp/vasp.6.4.2/bin:$PATH`
   - Or use full path: `/opt/vasp/vasp.6.4.2/bin/vasp_std`

5. OpenMPI help file warnings when testing:

   - Normal for `vasp_std --help` or similar test commands
   - VASP doesn't have standard help flags
   - Test with actual input files or use `timeout 5s vasp_std` to verify binary works

## Final Container Usage

**Run VASP calculations:**

```bash
# For production runs (VASP binaries are in PATH)
singularity exec --nv vasp-gpu.sif vasp_std

# Or specify full path
singularity exec --nv vasp-gpu.sif /opt/vasp/vasp.6.4.2/bin/vasp_std
```

**Development/compilation:**

```bash
# Mount working directory for data processing
singularity exec --bind ./data:/data vasp-gpu.sif python3 analysis_script.py
```

## Container Specifications
