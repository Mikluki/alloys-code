from pathlib import Path

from ase.io import read

target_dir = Path("data/AlCuNi_L1915_1400")
found = target_dir.glob("vasprun*")
# file = "vasprun.xml"
# vasprun_path = Path(target_dir, file)


for vasprun_path in found:
    print(vasprun_path)
    # minimal “does ASE read it?” test
    try:
        atoms0 = read(vasprun_path, index=0)  # first ionic step
        atoms1 = read(vasprun_path, index=1)  # second ionic step (fails if only 1)
    except Exception as e:
        print("ASE failed to read vasprun.xml:", repr(e))
        raise

    print("N atoms:", len(atoms0))
    print("Positions shape:", atoms0.positions.shape)

    v0 = atoms0.get_velocities()  # None if not present
    print("Velocities present:", v0 is not None)

    # to see how many steps ASE can read (may be heavier):
    images = read(vasprun_path, index=":")
    print("N ionic steps:", len(images))
    print()
