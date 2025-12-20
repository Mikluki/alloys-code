# VASP CPU Container Build Guide

## Quick setup

- **Step 1: Build sandbox first**

  ```bash
  sudo singularity build --sandbox vasp-cpu-dev-dir vasp-cpu.def
  ```

- **Step 2: Enter and test environment**

  ```bash
  sudo singularity shell --writable vasp-cpu-dev-dir

  # Test if Intel environment is automatically set
  which mpiifort        # Should show Intel MPI Fortran compiler
  which ifx            # Should show Intel Fortran compiler
  echo $MKLROOT        # Should show /opt/intel/oneapi/mkl/latest
  which vasp_std       # This should be empty since VASP isn't compiled yet

  # Test Intel compiler
  ifx --version        # Should show Intel Fortran compiler version

  # Test MKL linking
  echo $LD_LIBRARY_PATH | grep mkl  # Should show MKL libraries in path
  ```

- **Step 3: Compile VASP**

  ```bash
  cd /opt/vasp/vasp.6.4.3
  make DEPS=1 -j$(nproc)
  ```

  > Compilation time: ~8-12 minutes with Intel compilers (faster than GCC)

  > [!TIP] Expected behavior:
  >
  > - Dependency generation phase with Intel ifx
  > - Fortran compilation with AVX-512 optimization messages
  > - Linking with Intel MKL and Intel MPI libraries
  > - Creation of `vasp_std`, `vasp_gam`, `vasp_ncl` executables

- **Step 4: Check build and performance flags**

  ```bash
  # Re-enter the sandbox container where you compiled VASP
  singularity shell --writable vasp-cpu-dev-dir
  ls -la /opt/vasp/vasp.6.4.3/bin/

  # Check if executables are properly linked with Intel libraries
  ldd /opt/vasp/vasp.6.4.3/bin/vasp_std | grep -E "(intel|mkl)"

  # Quick startup test
  which vasp_std
  timeout 5s vasp_std 2>&1 | head -10
  ```

  This should show VASP starting up with Intel MKL initialization messages.

  > [!NOTE]
  > Performance verification:
  >
  > ```bash
  > # Check if AVX-512 optimization was applied
  > objdump -f /opt/vasp/vasp.6.4.3/bin/vasp_std | grep architecture
  >
  > # Check Intel library linkage
  > ldd /opt/vasp/vasp.6.4.3/bin/vasp_std | grep -E "(mkl|intel)"
  > ```

- **Step 5: Build final .sif**

  ```bash
  exit
  sudo singularity build vasp-cpu.sif vasp-cpu-dev-dir
  ```

## Resources

### Older intel toolkits

you can find links to older releases at [oneapi github](https://github.com/oneapi-src/oneapi-ci)

> [!NOTE] mpiifort, ifort icc icpc are only present in 2023 and below

```
LINUX_BASEKIT_URL: https://registrationcenter-download.intel.com/akdlm/IRC_NAS/992857b9-624c-45de-9701-f6445d845359/l_BaseKit_p_2023.2.0.49397_offline.sh
LINUX_HPCKIT_URL: https://registrationcenter-download.intel.com/akdlm/IRC_NAS/0722521a-34b5-4c41-af3f-d5d14e88248d/l_HPCKit_p_2023.2.0.49440_offline.sh
```

### Performance Notes

**Intel Compiler Advantages for VASP:**

- **AVX-512 vectorization**: ~2-3x speedup for FFTs and linear algebra
- **Intel MKL integration**: Optimized BLAS/LAPACK/SCALAPACK performance
- **Conservative optimization**: Official Intel makefile approach (-O2) for maximum reliability
- **Cross-compilation**: Targets cluster CPUs regardless of build machine hardware

**Target Hardware:**

- Optimized for: Intel Xeon Gold 6140 (Skylake-SP) - your cluster nodes
- Cross-compiled: Built on personal machine, optimized for cluster hardware
- Will work on: Most Intel CPUs with AVX-512 (2017+)
- Compatibility: Other Skylake, Cascade Lake, Ice Lake series

### Container Usage on Cluster

**Basic VASP run:**

```bash
# Single node run
singularity exec vasp-cpu.sif vasp_std

# Parallel with Slurm
sbatch --ntasks=1 --cpus-per-task=32 --wrap="singularity exec vasp-cpu-v2.sif mpirun -np 16 vasp_std"
```

> for this cluter slurm allocates threads, not cpus, thus cpus are set to x2

**Optimized NUMA binding for Xeon Gold 6140:**

```bash
# 4 NUMA domains, 9 cores each
mpirun --bind-to numa --map-by numa singularity exec vasp-cpu.sif vasp_std
```
