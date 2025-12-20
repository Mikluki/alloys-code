## K-Point Scaling Experiment Plan

### **Objective:**

Determine optimal k-point density vs CPU count trade-offs

### **Test Matrix:**

**K-point meshes:** 2×2×2, 3×3×3, 4×4×4, 5×5×5
**CPU configurations:** 8, 16, 32 cores  
**Total combinations:** 12 test cases

### **Test Structures:**

- Pick 3-5 representative structures from your thousands
- Use identical atomic configurations across all k-point/CPU tests
- Ensure structures represent typical random alloy diversity

### **Data Collection:**

For each combination, record:

- **Wall time** (total job duration)
- **CPU efficiency** (CPU time / wall time / cores)
- **Energy convergence** (compare final energies across k-point densities)

### **Implementation Steps:**

1. **Setup**: Create test directories with `-222`, `-333`, `-444`,`-555` suffixes
2. **Generate**: Run the KPOINTS script
3. **Submit**: Batch submit all 12×(number of structures) jobs
4. **Collect**: Parse OUTCAR files for timing/convergence data
5. **Analyze**: Plot scaling curves and identify optimal parameters

### **Expected Outcome:**

Identify the sweet spot where k-point accuracy meets computational efficiency for your specific alloy calculations.

**Timeline:** ~1-2 days for execution + analysis
