import logging
from pathlib import Path

from vsf.liquid.diagnostic.decorrelation import analyze_all_elements_decorrelation
from vsf.liquid.extract import extract_all_frames_from_vasprun
from vsf.liquid.workflows import (
    save_decorrelated_structures,
)
from vsf.logging import setup_logging

LOGGER = setup_logging(
    log_file=f"xliquid-correlate.log",
    console_level=logging.INFO,
)


base_dir = Path("a-md-Au300k")
output_decorr = Path("x-Au300k-decorr-poscar")
output_dir = Path("x-Au300k-analysis")
# base_dir = Path("tests-input")
# output_decorr = Path("x-tests")

# Extract all configs
configs_by_element = extract_all_frames_from_vasprun(base_dir=base_dir)

# Analyze correlation
decorr_results = analyze_all_elements_decorrelation(
    configs_by_element,
    frame_lag=10,
    delta=0.04,
    step_fallback=10,
    output_dir=output_dir,
)

save_decorrelated_structures(
    decorr_results,  # Now correct type
    output_dir=output_decorr,
    starting_id=3001500,
    prefix="decorr",
)
