# Outcome

- physical cpu -- primary resource <- `mpirun -np` askd for physical cores
- slurm on zhores allocates threads instead of physical cores
- on zhores 1 physical core != 1 thread.
- on zhores 1 physical 1 core == 1 threadh

## Cases

1.

```bash
sbatch --ntasks=16 --wrap="
source vsp-setup-cpuenv.sh &&
mpirun -np 16 --report-bindings vasp_std"

returns:

=== SLURM ENVIRONMENT ===
SLURM_NTASKS=16
SLURM_CPUS_PER_TASK=unset
SLURM_TASKS_PER_NODE=16
SLURM_CPUS_ON_NODE=16
SLURM_JOB_CPUS_PER_NODE=16
SLURM_NODELIST=ct04
=== NODE INFO ===
CPU(s): 160
On-line CPU(s) list: 0-159
Thread(s) per core: 2
Core(s) per socket: 20
Socket(s): 4
NUMA node0 CPU(s): 0-19,80-99
NUMA node1 CPU(s): 20-39,100-119
NUMA node2 CPU(s): 40-59,120-139
NUMA node3 CPU(s): 60-79,140-159

---

A request was made to bind to that would result in binding more
processes than cpus on a resource:

Bind to: CORE
Node: ct04
#processes: 2
#cpus: 1

You can override this protection by adding the "overload-allowed"
option to your binding directive.
```

2.

```bash
sbatch --ntasks=16 --wrap="
source vsp-setup-cpuenv.sh &&
mpirun -np 16 --bind-to hwthread --report-bindings vasp_std"

returns:

=== SLURM ENVIRONMENT ===
SLURM_NTASKS=16
SLURM_CPUS_PER_TASK=unset
SLURM_TASKS_PER_NODE=16
SLURM_CPUS_ON_NODE=16
SLURM_JOB_CPUS_PER_NODE=16
SLURM_NODELIST=ct04
=== NODE INFO ===
CPU(s):                160
On-line CPU(s) list:   0-159
Thread(s) per core:    2
Core(s) per socket:    20
Socket(s):             4
NUMA node0 CPU(s):     0-19,80-99
NUMA node1 CPU(s):     20-39,100-119
NUMA node2 CPU(s):     40-59,120-139
NUMA node3 CPU(s):     60-79,140-159
--------------------------------------------------------------------------
A request was made to bind to that would result in binding more
processes than cpus on a resource:

   Bind to:     HWTHREAD
   Node:        ct04
   #processes:  2
   #cpus:       1

You can override this protection by adding the "overload-allowed"
option to your binding directive.

```

3.

```bash
sbatch --ntasks=16 --cpus-per-task=2 --wrap="
source vsp-setup-cpuenv.sh &&
mpirun -np 16 --report-bindings vasp_std"

retunrs:

=== SLURM ENVIRONMENT ===
SLURM_NTASKS=16
SLURM_CPUS_PER_TASK=2
SLURM_TASKS_PER_NODE=16
SLURM_CPUS_ON_NODE=32
SLURM_JOB_CPUS_PER_NODE=32
SLURM_NODELIST=ct04
=== NODE INFO ===
CPU(s):                160
On-line CPU(s) list:   0-159
Thread(s) per core:    2
Core(s) per socket:    20
Socket(s):             4
NUMA node0 CPU(s):     0-19,80-99
NUMA node1 CPU(s):     20-39,100-119
NUMA node2 CPU(s):     40-59,120-139
NUMA node3 CPU(s):     60-79,140-159
[ct04.zhores:2075805] MCW rank 0 bound to socket 2[core 0[hwt 0]]: [][][B/./././././././././././.][././././././././././././././././././.]
[ct04.zhores:2075805] MCW rank 1 bound to socket 3[core 13[hwt 0]]: [][][././././././././././././.][B/./././././././././././././././././.]
[ct04.zhores:2075805] MCW rank 2 bound to socket 2[core 1[hwt 0]]: [][][./B/././././././././././.][././././././././././././././././././.]
[ct04.zhores:2075805] MCW rank 3 bound to socket 3[core 14[hwt 0]]: [][][././././././././././././.][./B/././././././././././././././././.]
[ct04.zhores:2075805] MCW rank 4 bound to socket 2[core 2[hwt 0]]: [][][././B/./././././././././.][././././././././././././././././././.]
[ct04.zhores:2075805] MCW rank 5 bound to socket 3[core 15[hwt 0]]: [][][././././././././././././.][././B/./././././././././././././././.]
[ct04.zhores:2075805] MCW rank 6 bound to socket 2[core 3[hwt 0]]: [][][./././B/././././././././.][././././././././././././././././././.]
[ct04.zhores:2075805] MCW rank 7 bound to socket 3[core 16[hwt 0]]: [][][././././././././././././.][./././B/././././././././././././././.]
[ct04.zhores:2075805] MCW rank 8 bound to socket 2[core 4[hwt 0]]: [][][././././B/./././././././.][././././././././././././././././././.]
[ct04.zhores:2075805] MCW rank 9 bound to socket 3[core 17[hwt 0]]: [][][././././././././././././.][././././B/./././././././././././././.]
[ct04.zhores:2075805] MCW rank 10 bound to socket 2[core 5[hwt 0]]: [][][./././././B/././././././.][././././././././././././././././././.]
[ct04.zhores:2075805] MCW rank 11 bound to socket 3[core 18[hwt 0]]: [][][././././././././././././.][./././././B/././././././././././././.]
[ct04.zhores:2075805] MCW rank 12 bound to socket 2[core 6[hwt 0]]: [][][././././././B/./././././.][././././././././././././././././././.]
[ct04.zhores:2075805] MCW rank 13 bound to socket 3[core 19[hwt 0]]: [][][././././././././././././.][././././././B/./././././././././././.]
[ct04.zhores:2075805] MCW rank 14 bound to socket 2[core 7[hwt 0]]: [][][./././././././B/././././.][././././././././././././././././././.]
[ct04.zhores:2075805] MCW rank 15 bound to socket 3[core 20[hwt 0]]: [][][././././././././././././.][./././././././B/././././././././././.]
 running   16 mpi-ranks, on    1 nodes
 distrk:  each k-point on    8 cores,    2 groups
 distr:  one band on    8 cores,    1 groups
 vasp.6.4.3 19Mar24 (build Jun 15 2025 02:10:56) complex
```

## Results

**Test 1 & 2 failed because:**

- `--ntasks=16` without `--cpus-per-task` → SLURM gave you 16 logical CPUs
- 16 logical CPUs = 8 physical cores (due to hyperthreading: 2 threads per core)
- MPI tried to place 16 processes on 8 cores → binding conflict
- Even `--bind-to hwthread` failed because SLURM's cgroup only exposed 16 logical CPUs to your job

**Test 3 worked because:**

- `--ntasks=16 --cpus-per-task=2` → SLURM gave you 32 logical CPUs
- 32 logical CPUs = 16 physical cores
- Now MPI has enough cores for 16 processes → no conflict

## Inference

The confusion is that "task" in SLURM doesn't mean "core" - it means "logical CPU" by default. Your VASP job needs 16 physical cores, so you need 32 logical CPUs on a hyperthreaded system.

I do not quite understand it. so slurm assign threads instead of physical cores and vasp explicitly requires physical cores?

### Solution

You need `--cpus-per-task=2` (or `--cpus-per-task=1 --hint=nomultithread`) to get enough physical cores for your 16 MPI processes.
