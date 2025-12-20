"""
GNN Runner - Subprocess executable for batch GNN energy calculations.

This module provides a standalone subprocess interface for calculating
formation energies using various GNN models (MACE, CHGNet, etc.) with
proper virtual environment isolation.

Usage:
    python -m vsf.gnn_runner --energy-source MACE --structure-list structures.txt

Key Features:
    - Batch processing with single calculator initialization
    - Error isolation - failed structures don't break the batch
    - Auto-generated failed structure files for reprocessing
    - Progress logging and detailed error reporting
    - Direct JSON file updates using existing StructureRecord methods
"""

__version__ = "1.0.0"

# from .factory import calculate_energy_per_atom, create_calculator
# from .interface import extract_calc_kwargs, parse_args
# from .processor import BatchProcessor, load_structure_list
#
# __all__ = [
#     "create_calculator",
#     "calculate_energy_per_atom",
#     "BatchProcessor",
#     "load_structure_list",
#     "parse_args",
#     "extract_calc_kwargs",
# ]
