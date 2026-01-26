import logging
from pathlib import Path
from vsf.core.hull.parser import parse_outcar, build_eos_table, eos_to_dataframe
from vsf.logging import setup_logging


LOGGER = setup_logging(
    console_level=logging.DEBUG,
    file_level=logging.DEBUG,
)

input_dir = Path("00-test")

# input_dir = Path("10-outcar-hull")

# path = Path("10-outcar-hull/Cu3Pd_mp-580357_0.8/OUTCAR")
# outcar_data = parse_outcar(path)
#
# print(outcar_data)

input_dir = Path("10-outcar-hull")
eos_points = build_eos_table(input_dir, method="vasp")
df = eos_to_dataframe(eos_points)
df.to_csv("eos_table.csv", index=False)
