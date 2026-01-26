#!/usr/bin/env python3
"""
Check geometry diffs between POSCAR and CONTCAR per directory.
```
python vsf-find-poscar-contcar-diff.py runs/mp-*/_* --pos-tol 1e-4 --cell-tol 1e-5
```
Prints one line per directory in this format:
<dir> :: CELL_DIFF :: <value_or_None> , ATOM_DIFF :: <value_or_None>

- CELL_DIFF is max abs difference of lattice matrix elements (Å)
- ATOM_DIFF is max PBC-aware atomic displacement (Å), computed from wrapped fractional
  coordinate differences and converted using the POSCAR lattice.

Errors are printed as:
<dir> :: ERROR :: <REASON>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Structure


def _has_glob(s: str) -> bool:
    return any(ch in s for ch in ["*", "?", "["])


def expand_input_dirs(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for a in args:
        p = Path(a).expanduser()
        if _has_glob(a):
            matches = sorted(Path().glob(a))
            out.extend([m for m in matches if m.is_dir()])
        else:
            out.append(p)
    # de-dup while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve() if p.exists() else p
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def species_sequence(struct: Structure) -> list[str]:
    # Keeps order: important if you want "did *this site* move"
    return [site.species_string for site in struct.sites]


def compute_diffs(pos: Structure, con: Structure) -> tuple[float, float]:
    # CELL_DIFF: max abs difference in lattice matrix elements (Angstrom)
    Lp = np.array(pos.lattice.matrix, dtype=float)
    Lc = np.array(con.lattice.matrix, dtype=float)
    cell_val = float(np.max(np.abs(Lc - Lp)))

    # ATOM_DIFF: max PBC-aware displacement in Angstrom
    fpos = np.array(pos.frac_coords, dtype=float)
    fcon = np.array(con.frac_coords, dtype=float)

    df = fcon - fpos
    df -= np.round(df)  # wrap into [-0.5, 0.5] per component

    # Convert fractional delta to cartesian delta using POSCAR lattice
    dr = df @ Lp  # (N,3) @ (3,3) -> (N,3)
    norms = np.linalg.norm(dr, axis=1)
    atom_val = float(np.max(norms)) if norms.size else 0.0

    return cell_val, atom_val


def fmt_val(x: float, tol: float) -> str:
    return "None" if x <= tol else f"{x:.6e}"


def process_dir(
    d: Path, cell_tol: float, pos_tol: float, dir_w: int, val_w: int
) -> str:
    d_str = str(d)

    poscar = d / "POSCAR"
    contcar = d / "CONTCAR"

    if not d.exists() or not d.is_dir():
        return f"{d_str:<{dir_w}} :: ERROR     :: NOT_A_DIR"
    if not poscar.exists():
        return f"{d_str:<{dir_w}} :: ERROR     :: MISSING_POSCAR"
    if not contcar.exists():
        return f"{d_str:<{dir_w}} :: ERROR     :: MISSING_CONTCAR"

    try:
        s_pos = Structure.from_file(poscar)
        s_con = Structure.from_file(contcar)
    except Exception as e:
        return f"{d_str:<{dir_w}} :: ERROR     :: PARSE_FAILED ({type(e).__name__})"

    if len(s_pos) != len(s_con):
        return f"{d_str:<{dir_w}} :: ERROR     :: N_ATOMS_MISMATCH ({len(s_pos)} vs {len(s_con)})"

    if species_sequence(s_pos) != species_sequence(s_con):
        return f"{d_str:<{dir_w}} :: ERROR     :: SPECIES_MISMATCH"

    cell_val, atom_val = compute_diffs(s_pos, s_con)

    cell_s = fmt_val(cell_val, cell_tol)
    atom_s = fmt_val(atom_val, pos_tol)

    return (
        f"{d_str:<{dir_w}} :: CELL_DIFF :: {cell_s:>{val_w}} , "
        f"ATOM_DIFF :: {atom_s:>{val_w}}"
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "dirs",
        nargs="+",
        help="Input directories (can include globs). Each dir must contain POSCAR and CONTCAR.",
    )
    ap.add_argument(
        "--cell-tol",
        type=float,
        default=1e-5,
        help="CELL_DIFF tolerance in Angstrom on lattice matrix elements (default: 1e-5).",
    )
    ap.add_argument(
        "--pos-tol",
        type=float,
        default=1e-4,
        help="ATOM_DIFF tolerance in Angstrom on max atomic displacement (default: 1e-3).",
    )

    ns = ap.parse_args(argv)

    dirs = expand_input_dirs(ns.dirs)
    if not dirs:
        print("No directories matched.", file=sys.stderr)
        return 2

    dir_w = max(len(str(d)) for d in dirs)
    val_w = 13  # fits '-1.234567e-03' (13 chars); 'None' will be padded

    for d in dirs:
        print(process_dir(d, ns.cell_tol, ns.pos_tol, dir_w, val_w))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
