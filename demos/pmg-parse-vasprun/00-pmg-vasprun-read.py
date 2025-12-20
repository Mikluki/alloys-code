from pymatgen.io.vasp.outputs import Outcar, Vasprun

if __name__ == "__main__":
    run_bad = "entries/Zr4_Ni4_ml-1000251_/vasprun.xml"
    run_good = "Zr4_Os4_ml-1000042_/vasprun.xml"

    vasprun_good = Vasprun(run_good, parse_dos=False, parse_eigen=False)
    print(f"### {vasprun_good.final_energy}")

    vasprun = Vasprun(run_bad, parse_dos=False, parse_eigen=False)
    print(f"### {vasprun.final_energy}")
