# Project A — Hull Relaxation Benchmark (EOS / Volume Scan)

## Prompt for new chat: Hull Relaxation Benchmarking (GNN vs VASP on 21-volume EOS scans)

I am working on a structured benchmarking experiment comparing GNN interatomic potentials against VASP on Materials Project structures that lie on or near the convex hull.

### Dataset context

For each MP structure, I have **21 volume-scaled variants** with volume factors:

`[0.80, 0.82, 0.84, ..., 1.20]`

For each volume scaling, VASP was run with:

- **ISIF = 2**, `IBRION = 1`, `NSW = 99`
- → atomic positions relaxed, **cell shape and volume held fixed**

I have VASP outputs containing:

- relaxed positions
- final energy
- final stress tensor
- hydrostatic pressure (via −⅓ Tr σ)

### GNN workflow

On the GNN side, I can reproduce the same protocol using my relaxation helper:

```python
relax_structure(
    structure_in,
    calculator=gnn,
    constant_cell_shape=True,
    constant_volume=True,
    scalar_pressure=0.0,
    fmax=0.02
)
```

This matches VASP’s ISIF=2 behavior.

### Goal of this chat

Your role is to function as a **high-level experiment designer**, not a code assistant.

I need help to:

1. **Audit what VASP EOS data I actually have** for each structure.
2. **Set up a clean GNN relaxation experiment** that mirrors the VASP protocol.
3. **Define the core metrics** (ΔP(V), ΔE(V), EOS fit parameters, etc.).
4. **Design the analysis pipeline** and prioritization: what to compute first, what comparisons matter, what plots demonstrate model behavior.
5. Keep me at the conceptual / structural level — assume I won’t paste code unless needed.

Task Zero: help me define a concise checklist to validate what each VASP dataset contains before running GNN relaxations.

# Project B — MD Snapshot Benchmarking (High-T Local Force/Stress Test)

I want to analyze a small VASP molecular dynamics dataset, but I am deliberately starting with **zero assumptions** about what the files contain. I may have partial OUTCARs, an XDATCAR, or other fragments, and I don’t know yet what information is actually present or usable.

Your role in this chat is to act as a **high-level diagnostic guide**, not a code debugging assistant.

I will describe:

* which files exist (OUTCAR, XDATCAR, vasprun.xml, etc.),

Your job is to help me:

1. **Audit the dataset from scratch** — determine what information truly exists, what is missing, and what cannot be recovered.
2. **Clarify the structure of an MD run in VASP** (ionic steps, energies, forces, stresses, positions), so we can map my files onto the expected format.
3. **Define a clean set of possible analyses** based on what we actually have (e.g., GNN vs VASP forces on available frames), *without* assuming that full MD trajectories exist.
4. **Stop me from pursuing analyses that the data cannot support** (e.g., RDF or MSD if I don’t have enough frames).
5. After the audit, help me design a **minimal, meaningful benchmarking experiment** using only the data that actually exists.

Important constraints:

* I may not provide code unless absolutely necessary.
* My descriptions will be factual rather than detailed dumps.
* Your guidance should stay at a conceptual, structural level — think “scientific investigator,” not “programmer.”
* Treat missing or partial data as expected; help me navigate uncertainty calmly and logically.

Let’s begin with **Task Zero**:
Help me construct a short checklist for assessing what MD-related information each file (OUTCAR, XDATCAR, vasprun.xml, etc.) contains, starting from no assumptions.

