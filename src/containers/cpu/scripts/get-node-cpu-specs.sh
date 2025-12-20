#!/bin/bash
#SBATCH --job-name=cpu_check
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=cpu_info_%j.out
#SBATCH --error=cpu_info_%j.err

echo "========================================="
echo "CPU Information Check on Compute Node"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "========================================="
echo $SLURMD_NODENAME

echo -e "\n=== BASIC CPU INFO ==="
lscpu | grep -E "(Model name|Flags|Architecture|CPU\(s\)|Thread|Core|Socket|MHz|Cache)"

echo -e "\n=== DETAILED CPU MODEL ==="
cat /proc/cpuinfo | grep -E "(model name|flags|cpu MHz|cache size)" | head -8

echo -e "\n=== VECTOR INSTRUCTION SUPPORT ==="
echo "Available SIMD instructions:"
grep flags /proc/cpuinfo | head -1 | grep -o 'sse[^ ]*\|avx[^ ]*\|fma[^ ]*\|avx512[^ ]*' | sort | uniq | tr '\n' ' '
echo

echo -e "\n=== NUMA TOPOLOGY ==="
lscpu | grep -E "(NUMA|node)"

echo -e "\n=== MEMORY INFO ==="
free -h | head -2

echo -e "\n=== GCC MARCH NATIVE DETECTION (if available) ==="
if command -v gcc &> /dev/null; then
    echo "GCC version: $(gcc --version | head -1)"
    echo "GCC -march=native would use:"
    gcc -march=native -Q --help=target 2>/dev/null | grep -E "(march|mtune)" | head -5 || echo "GCC feature detection not available"
else
    echo "GCC not available on compute node"
fi

echo -e "\n=== INTEL CPU FEATURES (detailed) ==="
# Check for specific Intel generations/features
CPUMODEL=$(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
echo "CPU Model: $CPUMODEL"

# Check Intel generation
if echo "$CPUMODEL" | grep -qi "cascade"; then
    echo "Intel Generation: Cascade Lake (2019+)"
elif echo "$CPUMODEL" | grep -qi "skylake\|gold\|platinum.*[^0-9][89][0-9][0-9][0-9]"; then
    echo "Intel Generation: Skylake-SP (2017+)"
elif echo "$CPUMODEL" | grep -qi "broadwell"; then
    echo "Intel Generation: Broadwell (2014+)"
elif echo "$CPUMODEL" | grep -qi "haswell"; then
    echo "Intel Generation: Haswell (2013+)"
else
    echo "Intel Generation: Unknown, check model manually"
fi

echo -e "\n=== RECOMMENDED VASP COMPILER FLAGS ==="
FLAGS=$(grep flags /proc/cpuinfo | head -1)

if echo "$FLAGS" | grep -q "avx512f"; then
    echo "✓ AVX-512 supported - can use aggressive optimization"
    echo "Recommended: -march=skylake-avx512 or -march=core-avx512"
elif echo "$FLAGS" | grep -q "avx2"; then
    echo "✓ AVX2 supported - good performance"
    echo "Recommended: -march=core-avx2"
elif echo "$FLAGS" | grep -q "avx"; then
    echo "✓ AVX supported - basic optimization"
    echo "Recommended: -march=core-avx-i"
else
    echo "! Limited vector support"
    echo "Recommended: -march=core2"
fi

echo -e "\n=== SLURM ENVIRONMENT ==="
echo "Partition: $SLURM_JOB_PARTITION"
echo "QOS: $SLURM_JOB_QOS"
echo "CPUs allocated: $SLURM_CPUS_ON_NODE"

echo -e "\n========================================="
echo "CPU check completed on: $(date)"
echo "========================================="
