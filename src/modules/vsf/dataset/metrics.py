"""
GNN Evaluation Pipeline
Processes structure JSONs, extracts predictions, and computes evaluation metrics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from sklearn.metrics import mean_absolute_error, r2_score

LOGGER = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATTERNS = {
    # "pure": ["-pure-"],
    "binary_alloys": ["-cat-"],
    "liquid_alloys": ["-liquid-"],
    "random_alloys": ["-rand-"],
    "convex_hull": ["-hull-"],
}

GNN_NAMES = [
    "ORBV3",
    "SEVENNET",
    "ESEN_30M_OAM",
    "MACE_MPA_0",
    "NEQUIP",
    "ALLEGRO",
    "DPA31",
    "GRACE_2L_OAM_L",
]

DEFAULT_STABILITY_THRESHOLD = 0.010


# ============================================================================
# STRESS HANDLING
# ============================================================================


def _extract_pressure_from_stress(stress_array: List[float]) -> float:
    """
    Convert 6-component stress tensor to hydrostatic pressure.

    The stress array is in ASE Voigt notation (xx, yy, zz, yz, xz, xy).
    Only the diagonal components (xx, yy, zz) are used for pressure.

    Args:
        stress_array: List of 6 floats representing stress tensor in eV/Å³

    Returns:
        float: Hydrostatic pressure in eV/Å³

    Raises:
        ValueError: If stress_array length != 6 or components are non-numeric

    Notes:
        - Input units: eV/Å³ (from VASP OUTCAR via ASE parser)
        - Output units: eV/Å³
        - Sign convention: pressure = -(σ_xx + σ_yy + σ_zz) / 3.0
        - Positive pressure = compression
    """
    if len(stress_array) != 6:
        raise ValueError(
            f"Expected 6-component stress array, got {len(stress_array)}: {stress_array}"
        )

    try:
        stress_floats = [float(s) for s in stress_array]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Could not convert stress components to float: {stress_array}"
        ) from e

    pressure = -(stress_floats[0] + stress_floats[1] + stress_floats[2]) / 3.0
    return pressure


# ============================================================================
# DATA LOADING
# ============================================================================


def _load_structure_json(filepath: Path) -> Dict[str, Any]:
    """Load a single structure JSON file."""
    LOGGER.info(f"Loading: {filepath.name}")
    with open(filepath, "r") as f:
        return json.load(f)


def load_structures_from_directory(directory: Path) -> List[Dict[str, Any]]:
    """Load all JSON structures from a directory."""
    structures = []
    json_files = list(directory.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {directory}")

    for filepath in json_files:
        structure = _load_structure_json(filepath)
        structures.append(structure)

    LOGGER.debug(f"Loaded {len(structures)} structures from {directory}")
    return structures


def _identify_dataset(directory_path: Path) -> str:
    """Identify dataset name from directory path using pattern matching."""
    path_str = str(directory_path)

    matched_datasets = []
    for dataset_name, patterns in DATASET_PATTERNS.items():
        if any(pattern in path_str for pattern in patterns):
            matched_datasets.append(dataset_name)

    if len(matched_datasets) == 0:
        raise ValueError(f"No dataset pattern matched for directory: {directory_path}")

    if len(matched_datasets) > 1:
        raise ValueError(
            f"Multiple dataset patterns matched for directory: {directory_path}\n"
            f"Matched: {matched_datasets}"
        )

    return matched_datasets[0]


# ============================================================================
# METADATA EXTRACTION
# ============================================================================


def _extract_metadata(structure: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from a structure."""
    vasp_formation = structure["formation_energy"]["results"]["VASP"]["value"]
    vasp_potential = structure["potential_energy"]["results"]["VASP"]["value"]

    metadata = {
        "formula": structure["formula"],
        "reduced_formula": structure["reduced_formula"],
        "atoms": structure["atoms"],
        "num_sites": structure["num_sites"],
        "vasp_formation_energy": vasp_formation,
        "vasp_potential_energy": vasp_potential,
    }

    return metadata


def _build_metadata_dict(structures: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build metadata dictionary for all structures."""
    metadata_dict = {}

    for structure in structures:
        structure_name = structure["name"]
        metadata_dict[structure_name] = _extract_metadata(structure)

    return metadata_dict


# ============================================================================
# PREDICTION EXTRACTION
# ============================================================================


def _extract_predictions(
    structures: List[Dict[str, Any]], dataset_name: str, stability_threshold: float
) -> List[Dict[str, Any]]:
    """
    Extract all predictions from structures.

    Handles three property types:
    - formation_energy: scalar predictions with stability classification
    - potential_energy: scalar predictions (regression only)
    - stress_analyzer: stress tensors converted to hydrostatic pressure
    """
    predictions = []

    for structure in structures:
        structure_name = structure["name"]

        vasp_formation = structure["formation_energy"]["results"]["VASP"]["value"]

        for gnn_name in GNN_NAMES:
            # ================================================================
            # Formation Energy
            # ================================================================
            if gnn_name in structure["formation_energy"]["results"]:
                gnn_formation = structure["formation_energy"]["results"][gnn_name][
                    "value"
                ]

                vasp_stable = vasp_formation <= stability_threshold
                gnn_stable = gnn_formation <= stability_threshold

                predictions.append(
                    {
                        "dataset": dataset_name,
                        "structure_name": structure_name,
                        "gnn_name": gnn_name,
                        "property": "formation_energy",
                        "vasp_value": vasp_formation,
                        "gnn_value": gnn_formation,
                        "vasp_stable": vasp_stable,
                        "gnn_stable": gnn_stable,
                        "classification_match": vasp_stable == gnn_stable,
                        "stability_threshold": stability_threshold,
                    }
                )
            else:
                LOGGER.warning(
                    f"`{gnn_name}` not found for `{structure_name}` in `{dataset_name}` (formation_energy)"
                )

            # ================================================================
            # Stress Analyzer → Pressure
            # ================================================================
            if "stress_analyzer" in structure:
                if gnn_name in structure["stress_analyzer"]["results"]:
                    try:
                        stress_array = structure["stress_analyzer"]["results"][
                            gnn_name
                        ]["stress_array"]
                        gnn_pressure = _extract_pressure_from_stress(stress_array)

                        # Extract VASP pressure (same structure as GNN)
                        if "VASP" in structure["stress_analyzer"]["results"]:
                            vasp_stress_array = structure["stress_analyzer"]["results"][
                                "VASP"
                            ]["stress_array"]
                            vasp_pressure = _extract_pressure_from_stress(
                                vasp_stress_array
                            )

                            predictions.append(
                                {
                                    "dataset": dataset_name,
                                    "structure_name": structure_name,
                                    "gnn_name": gnn_name,
                                    "property": "stress_pressure",
                                    "vasp_value": vasp_pressure,
                                    "gnn_value": gnn_pressure,
                                }
                            )
                        else:
                            LOGGER.error(
                                f"VASP stress missing for `{structure_name}` in `{dataset_name}`"
                            )

                    except (KeyError, ValueError) as e:
                        LOGGER.error(
                            f"Failed to extract stress for `{gnn_name}` in `{structure_name}` (`{dataset_name}`): {e}"
                        )
                        continue
                else:
                    LOGGER.error(
                        f"Stress data missing for `{gnn_name}` in `{structure_name}` (`{dataset_name}`)"
                    )

    return predictions


# ============================================================================
# METRICS CALCULATION
# ============================================================================


def _calculate_regression_metrics(
    vasp_values: List[float], gnn_values: List[float]
) -> Dict[str, float]:
    """Calculate regression metrics (MAE and R²)."""
    mae = mean_absolute_error(vasp_values, gnn_values)
    r_squared = r2_score(vasp_values, gnn_values)

    return {"mae": mae, "r_squared": r_squared}


def _calculate_classification_metrics(
    vasp_values: List[float], gnn_values: List[float], threshold: float
) -> Dict[str, Any]:
    """Calculate classification metrics for stability prediction."""
    vasp_stable = [v <= threshold for v in vasp_values]
    gnn_stable = [v <= threshold for v in gnn_values]

    true_positive = sum(1 for v, g in zip(vasp_stable, gnn_stable) if v and g)
    false_positive = sum(1 for v, g in zip(vasp_stable, gnn_stable) if not v and g)
    false_negative = sum(1 for v, g in zip(vasp_stable, gnn_stable) if v and not g)
    true_negative = sum(1 for v, g in zip(vasp_stable, gnn_stable) if not v and not g)

    total = len(vasp_values)
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    accuracy = (true_positive + true_negative) / total if total > 0 else 0.0

    return {
        "true_positive_count": true_positive,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "true_negative_count": true_negative,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def _compute_aggregated_metrics(
    predictions: List[Dict[str, Any]], stability_threshold: float
) -> List[Dict[str, Any]]:
    """Compute aggregated metrics grouped by dataset, GNN, and property."""
    aggregated = []

    # Group predictions by (dataset, gnn_name, property)
    groups = {}
    for pred in predictions:
        key = (pred["dataset"], pred["gnn_name"], pred["property"])
        if key not in groups:
            groups[key] = []
        groups[key].append(pred)

    # Calculate metrics for each group
    for (dataset, gnn_name, property_name), group_preds in groups.items():
        vasp_values = [p["vasp_value"] for p in group_preds]
        gnn_values = [p["gnn_value"] for p in group_preds]

        # Regression metrics (for all properties)
        regression_metrics = _calculate_regression_metrics(vasp_values, gnn_values)

        result = {
            "dataset": dataset,
            "gnn_name": gnn_name,
            "property": property_name,
            "n_structures": len(group_preds),
            "mae": regression_metrics["mae"],
            "r_squared": regression_metrics["r_squared"],
        }

        # Classification metrics (only for formation_energy)
        if property_name == "formation_energy":
            classification_metrics = _calculate_classification_metrics(
                vasp_values, gnn_values, stability_threshold
            )
            result.update(
                {"stability_threshold": stability_threshold, **classification_metrics}
            )

        aggregated.append(result)

    return aggregated


# ============================================================================
# FILE I/O
# ============================================================================


def _save_json(data: Any, filepath: Path):
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    LOGGER.info(f"Saved: {filepath}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def process_datasets(
    base_directories: List[Path],
    output_dir: Path,
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
):
    """
    Process datasets from base directories containing structure subdirectories.

    Args:
        base_directories: List of base dataset directories (e.g., rand-eform/e-rand-1-gnncp)
        output_dir: Directory to save output files
        stability_threshold: Threshold for stability classification (eV/atom)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Starting GNN Evaluation Pipeline")
    LOGGER.info(f"Stability threshold: {stability_threshold} eV/atom")
    LOGGER.info(f"{'='*60}\n")

    all_structures = []
    all_predictions = []

    for base_dir in base_directories:
        # Identify dataset once per base directory
        dataset_name = _identify_dataset(base_dir)
        LOGGER.info(f"Dataset: `{dataset_name}` from {base_dir}")

        # Find all structure subdirectories
        structure_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

        # Load structures from all subdirectories
        for structure_dir in structure_dirs:
            structures = load_structures_from_directory(structure_dir)
            predictions = _extract_predictions(
                structures, dataset_name, stability_threshold
            )

            for structure in structures:
                structure["_dataset"] = dataset_name

            all_structures.extend(structures)
            all_predictions.extend(predictions)

        dataset_count = len(
            [s for s in all_structures if s["_dataset"] == dataset_name]
        )
        LOGGER.info(f"  Loaded {dataset_count} structures")

    LOGGER.info("")
    LOGGER.info(f"Total structures loaded: {len(all_structures)}")
    LOGGER.info(f"Total predictions extracted: {len(all_predictions)}\n")

    # Step 2: Build and save metadata
    LOGGER.info("Building metadata...")
    metadata_dict = _build_metadata_dict(all_structures)
    _save_json(metadata_dict, output_dir / "structure_metadata.json")

    # Step 3: Save predictions
    LOGGER.info("Saving predictions...")
    _save_json(all_predictions, output_dir / "structure_predictions.json")

    # Step 4: Compute and save aggregated metrics
    LOGGER.info("Computing aggregated metrics...")
    aggregated_metrics = _compute_aggregated_metrics(
        all_predictions, stability_threshold
    )

    threshold_str = str(stability_threshold).replace(".", "")
    metrics_filename = f"aggregated_metrics_T{threshold_str}.json"
    _save_json(aggregated_metrics, output_dir / metrics_filename)

    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"Pipeline completed successfully!")
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"{'='*60}\n")
