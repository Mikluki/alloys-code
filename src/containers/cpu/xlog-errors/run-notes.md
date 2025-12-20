### Force TCP/SHM Transport, srun

```bash
sbatch --ntasks=16 --cpus-per-task=2 --wrap="
srun singularity exec --env I_MPI_FABRICS=shm:tcp --env I_MPI_HYDRA_BOOTSTRAP=fork --env I_MPI_PMI=no /beegfs/home/p.zhilyaev/mklk/apps/vasp-cpu.sif vasp_std"
```

```bash
Warning: Environment variable I_MPI_HYDRA_BOOTSTRAP already has value [fork], will not forward new value [slurm] from parent process environment
MPI startup(): PMI server not found. Please set I_MPI_PMI_LIBRARY variable if it is not a singleton case.
Abort(672785807) on node 0 (rank 0 in comm 0): Fatal error in internal_Init: Other MPI error, error stack:
internal_Init(39850).........: MPI_Init(argc=(nil), argv=(nil)) failed
MPII_Init_thread(118)........:
MPID_Init(1719)..............:
MPIDI_OFI_mpi_init_hook(1661):
(unknown)(): Other MPI error
[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=672785807
:
system msg for write_line failure : Bad file descriptor
Abort(672785807) on node 0 (rank 0 in comm 0): Fatal error in internal_Init: Other MPI error, error stack:
internal_Init(39850).........: MPI_Init(argc=(nil), argv=(nil)) failed
MPII_Init_thread(118)........:
MPID_Init(1719)..............:
MPIDI_OFI_mpi_init_hook(1661):
(unknown)(): Other MPI error
[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=672785807
:
system msg for write_line failure : Bad file descriptor
forrtl: severe (174): SIGSEGV, segmentation fault occurred
Image              PC                Routine            Line        Source
libc.so.6          00001555439F0520  Unknown               Unknown  Unknown
libmpi.so.12.0.0   0000155544467D0B  MPIR_Err_return_c     Unknown  Unknown
libmpifort.so.12.  000015554E32857F  MPI_INIT              Unknown  Unknown
vasp_std           000000000043CA66  Unknown               Unknown  Unknown
vasp_std           000000000048A358  Unknown               Unknown  Unknown
vasp_std           00000000019DD219  Unknown               Unknown  Unknown
vasp_std           000000000040B5ED  Unknown               Unknown  Unknown
libc.so.6          00001555439D7D90  Unknown               Unknown  Unknown
libc.so.6          00001555439D7E40  __libc_start_main     Unknown  Unknown
vasp_std           000000000040B505  Unknown               Unknown  Unknown
MPI startup(): PMI server not found. Please set I_MPI_PMI_LIBRARY variable if it is not a singleton case.
```

### Quick Test: Serial Run First

```bash
sbatch --ntasks=1 --cpus-per-task=1 --wrap="singularity exec /beegfs/home/p.zhilyaev/mklk/apps/vasp-cpu.sif vasp_std"
```

```bash
Abort(672785807) on node 0 (rank 0 in comm 0): Fatal error in internal_Init: Other MPI error, error stack:
internal_Init(39850).........: MPI_Init(argc=(nil), argv=(nil)) failed
MPII_Init_thread(118)........:
MPID_Init(1719)..............:
MPIDI_OFI_mpi_init_hook(1661):
(unknown)(): Other MPI error
[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=672785807
:
system msg for write_line failure : Bad file descriptor
Abort(672785807) on node 0 (rank 0 in comm 0): Fatal error in internal_Init: Other MPI error, error stack:
internal_Init(39850).........: MPI_Init(argc=(nil), argv=(nil)) failed
MPII_Init_thread(118)........:
MPID_Init(1719)..............:
MPIDI_OFI_mpi_init_hook(1661):
(unknown)(): Other MPI error
[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=672785807
:
system msg for write_line failure : Bad file descriptor
forrtl: severe (174): SIGSEGV, segmentation fault occurred
Image              PC                Routine            Line        Source
libc.so.6          00001555439F0520  Unknown               Unknown  Unknown
libmpi.so.12.0.0   0000155544467D0B  MPIR_Err_return_c     Unknown  Unknown
libmpifort.so.12.  000015554E32857F  MPI_INIT              Unknown  Unknown
vasp_std           000000000043CA66  Unknown               Unknown  Unknown
vasp_std           000000000048A358  Unknown               Unknown  Unknown
vasp_std           00000000019DD219  Unknown               Unknown  Unknown
vasp_std           000000000040B5ED  Unknown               Unknown  Unknown
libc.so.6          00001555439D7D90  Unknown               Unknown  Unknown
libc.so.6          00001555439D7E40  __libc_start_main     Unknown  Unknown
vasp_std           000000000040B505  Unknown               Unknown  Unknown

```

### Single Task + Internal MPI

```bash
sbatch --ntasks=1 --cpus-per-task=32 --wrap="
singularity exec --env I_MPI_FABRICS=shm:tcp --env I_MPI_HYDRA_BOOTSTRAP=fork /beegfs/home/p.zhilyaev/mklk/apps/vasp-cpu.sif mpirun -np 16 vasp_std"
```

```bash
Warning: Environment variable I_MPI_HYDRA_BOOTSTRAP already has value [fork], will not forward new value [slurm] from parent process environment
Abort(672785807) on node 0 (rank 0 in comm 0): Fatal error in internal_Init: Other MPI error, error stack:
internal_Init(39850).........: MPI_Init(argc=(nil), argv=(nil)) failed
MPII_Init_thread(118)........:
MPID_Init(1719)..............:
MPIDI_OFI_mpi_init_hook(1661):
(unknown)(): Other MPI error
```

### Stronger MPI Isolation srun

```bash
sbatch --ntasks=16 --cpus-per-task=2 --wrap="srun singularity exec --env I_MPI_FABRICS=shm:tcp --env I_MPI_HYDRA_BOOTSTRAP=fork --env I_MPI_PMI=no --env I_MPI_HYDRA_EXEC=fork --env SLURM_MPI_TYPE=none /beegfs/home/p.zhilyaev/mklk/apps/vasp-cpu.sif vasp_std"
```

```bash
Warning: Environment variable I_MPI_HYDRA_BOOTSTRAP already has value [fork], will not forward new value [slurm] from parent process environment
MPI startup(): PMI server not found. Please set I_MPI_PMI_LIBRARY variable if it is not a singleton case.
MPI startup(): PMI server not found. Please set I_MPI_PMI_LIBRARY variable if it is not a singleton case.
Abort(672785807) on node 0 (rank 0 in comm 0): Fatal error in internal_Init: Other MPI error, error stack:
internal_Init(39850).........: MPI_Init(argc=(nil), argv=(nil)) failed
MPII_Init_thread(118)........:
MPID_Init(1719)..............:
MPIDI_OFI_mpi_init_hook(1661):
(unknown)(): Other MPI error
[unset]: write_line error; fd=-1 buf=:cmd=abort exitcode=672785807
:
MPI startup(): PMI server not found. Please set I_MPI_PMI_LIBRARY variable if it is not a singleton case.
forrtl: severe (174): SIGSEGV, segmentation fault occurred
Image              PC                Routine            Line        Source
libc.so.6          00001555439F0520  Unknown               Unknown  Unknown
libmpi.so.12.0.0   0000155544467D0B  MPIR_Err_return_c     Unknown  Unknown
libmpifort.so.12.  000015554E32857F  MPI_INIT              Unknown  Unknown
vasp_std           000000000043CA66  Unknown               Unknown  Unknown
vasp_std           000000000048A358  Unknown               Unknown  Unknown
vasp_std           00000000019DD219  Unknown               Unknown  Unknown
vasp_std           000000000040B5ED  Unknown               Unknown  Unknown
libc.so.6          00001555439D7D90  Unknown               Unknown  Unknown
libc.so.6          00001555439D7E40  __libc_start_main     Unknown  Unknown
vasp_std           000000000040B505  Unknown               Unknown  Unknown
forrtl: severe (174): SIGSEGV, segmentation fault occurred
```

## mpi debug

### Step 1: Test Intel MPI directly (not VASP)

```bash
singularity exec /beegfs/home/p.zhilyaev/mklk/apps/vasp-cpu.sif mpirun --version
Intel(R) MPI Library for Linux* OS, Version 2021.16 Build 20250513 (id: a7c135c)
Copyright 2003-2025, Intel Corporation.
```

```bash
singularity exec /beegfs/home/p.zhilyaev/mklk/apps/vasp-cpu.sif mpirun -np 1 echo "MPI test"
MPI test
```

### Step 2: Check Intel MPI environment

```bash
# Check what's available inside container
singularity exec /beegfs/home/p.zhilyaev/mklk/apps/vasp-cpu.sif bash -c "
> echo 'Intel MPI version:'; mpirun --version
> echo 'Available fabrics:'; ls /opt/intel/oneapi/mpi/*/lib/release/ | grep fabric
> echo 'Library path:'; echo \$LD_LIBRARY_PATH | tr ':' '\n' | grep -E 'mpi|intel'
> "
Intel MPI version:
Intel(R) MPI Library for Linux* OS, Version 2021.16 Build 20250513 (id: a7c135c)
Copyright 2003-2025, Intel Corporation.
Available fabrics:
Library path:
/opt/intel/oneapi/mpi/latest/lib
/opt/intel/oneapi/compiler/latest/lib
/opt/intel/oneapi/mkl/latest/lib

```

### Option 3: Check VASP compilation

```bash
singularity shell /beegfs/home/p.zhilyaev/mklk/apps/vasp-cpu.sif
Singularity> ldd /opt/vasp/vasp.6.4.3/bin/vasp_std | grep mpi
        libmkl_blacs_intelmpi_lp64.so.2 => /opt/intel/oneapi/mkl/latest/lib/libmkl_blacs_intelmpi_lp64.so.2 (0x0000150db5607000)
        libmpifort.so.12 => /opt/intel/oneapi/mpi/latest/lib/libmpifort.so.12 (0x0000150db5200000)
        libmpi.so.12 => /opt/intel/oneapi/mpi/latest/lib/libmpi.so.12 (0x0000150dab000000)
        libimf.so => /opt/intel/oneapi/compiler/latest/lib/libimf.so (0x0000150daabd7000)
        libintlc.so.5 => /opt/intel/oneapi/compiler/latest/lib/libintlc.so.5 (0x0000150daa94c000)
```

## mpi 2

### Step 1: Test VASP Serial Execution (if available)

```bash
# Check if you have a serial VASP version
singularity shell /beegfs/home/p.zhilyaev/mklk/apps/vasp-cpu.sif
Singularity> ls -la /opt/vasp/vasp.6.4.3/bin/
total 87714
-rwxr-xr-x 1 p.zhilyaev p.zhilyaev 29690680 Jun 26 19:18 vasp_gam
-rwxr-xr-x 1 p.zhilyaev p.zhilyaev 30093224 Jun 26 19:26 vasp_ncl
-rwxr-xr-x 1 p.zhilyaev p.zhilyaev 30035864 Jun 26 19:11 vasp_std
```

### Step 2: Force Conservative MPI Settings for VASP

```bash
singularity exec --env I_MPI_FABRICS=tcp --env FI_PROVIDER=tcp --env I_MPI_FALLBACK=1 /beegfs/home/p.zhilyaev/mklk/apps/vasp-cpu.sif mpirun -np 1 vasp_std
Abort(672785807) on node 0 (rank 0 in comm 0): Fatal error in internal_Init: Other MPI error, error stack:
internal_Init(39850).........: MPI_Init(argc=(nil), argv=(nil)) failed
MPII_Init_thread(118)........:
MPID_Init(1719)..............:
MPIDI_OFI_mpi_init_hook(1661):
(unknown)(): Other MPI error
```

This one caused due to missing `libfabric1`

### Step 3: Check VASP Compilation Flags

```bash
Singularity> cat /opt/vasp/vasp.6.4.3/makefile.include
# Default precompiler options
CPP_OPTIONS = -DHOST=\"LinuxIFC\" \
              -DMPI -DMPI_BLOCK=8000 -Duse_collective \
              -DscaLAPACK \
              -DCACHE_SIZE=4000 \
              -Davoidalloc \
              -Dvasp6 \
              -Duse_bse_te \
              -Dtbdyn \
              -Dfock_dblbuf

CPP         = fpp -f_com=no -free -w0  $*$(FUFFIX) $*$(SUFFIX) $(CPP_OPTIONS)

FC          = mpiifx
FCL         = mpiifx

FREE        = -free -names lowercase

FFLAGS      = -assume byterecl -w

OFLAG       = -O2
OFLAG_IN    = $(OFLAG)
DEBUG       = -O0

OBJECTS     = fftmpiw.o fftmpi_map.o fftw3d.o fft3dlib.o
OBJECTS_O1 += fftw3d.o fftmpi.o fftmpiw.o
OBJECTS_O2 += fft3dlib.o

# For what used to be vasp.5.lib
CPP_LIB     = $(CPP)
FC_LIB      = $(FC)
CC_LIB      = icx
CFLAGS_LIB  = -O
FFLAGS_LIB  = -O1
FREE_LIB    = $(FREE)

OBJECTS_LIB = linpack_double.o

# For the parser library
CXX_PARS    = icpx
LLIBS       = -lstdc++

##
## Customize as of this point! Of course you may change the preceding
## part of this file as well if you like, but it should rarely be
## necessary ...
##

# When compiling on the target machine itself, change this to the
# relevant target when cross-compiling for another architecture
# Targeting Intel Xeon Gold 6140 (Skylake-SP) on cluster nodes
VASP_TARGET_CPU ?= -xCORE-AVX512
FFLAGS     += $(VASP_TARGET_CPU)

# Intel MKL (FFTW, BLAS, LAPACK, and scaLAPACK)
# (Note: for Intel Parallel Studio's MKL use -mkl instead of -qmkl)
FCL        += -qmkl=sequential
MKLROOT    ?= /opt/intel/oneapi/mkl/latest
LLIBS      += -L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
INCS        =-I$(MKLROOT)/include/fftw

# HDF5-support (optional but strongly recommended)
# Disabled for this build - can be enabled later if needed
#CPP_OPTIONS+= -DVASP_HDF5
#HDF5_ROOT  ?= /path/to/your/hdf5/installation
#LLIBS      += -L$(HDF5_ROOT)/lib -lhdf5_fortran
#INCS       += -I$(HDF5_ROOT)/include

# For the VASP-2-Wannier90 interface (optional)
#CPP_OPTIONS    += -DVASP2WANNIER90
#WANNIER90_ROOT ?= /path/to/your/wannier90/installation
#LLIBS          += -L$(WANNIER90_ROOT)/lib -lwannier
```

## install libfabric

I rebuild container with libfabric installed

```bash
sbatch --ntasks=16 --cpus-per-task=2 --wrap="srun singularity exec vasp-cpu-v2.sif vasp_std"
```

in logs it runs!:

```bash
MPI startup(): PMI server not found. Please set I_MPI_PMI_LIBRARY variable if it is not a singleton case.
 running    1 mpi-ranks, on    1 nodes
 running    1 mpi-ranks, on    1 nodes
 running    1 mpi-ranks, on    1 nodes
 distrk:  each k-point on    1 cores,    1 groups
 distr:  one band on    1 cores,    1 groups
 distrk:  each k-point on    1 cores,    1 groups
 distr:  one band on    1 cores,    1 groups
 distrk:  each k-point on    1 cores,    1 groups
 distr:  one band on    1 cores,    1 groups
 vasp.6.4.3 19Mar24 (build Jun 26 2025 15:56:33) complex

 vasp.6.4.3 19Mar24 (build Jun 26 2025 15:56:33) complex
```

However as you can see parallelizaion is none

## parallelizaion

```bash
sbatch --ntasks=1 --cpus-per-task=32 --wrap="singularity exec vasp-cpu-v2.sif mpirun -np 16 vasp_std"
```

logs:

```bash
vasp.6.4.3 19Mar24 (build Jun 15 2025 02:10:56) complex

executed on             LinuxIFC date 2025.06.16  17:42:48
running   16 mpi-ranks, on    1 nodes
distrk:  each k-point on   16 cores,    1 groups
distr:  one band on NCORE=  16 cores,    1 groups
```
