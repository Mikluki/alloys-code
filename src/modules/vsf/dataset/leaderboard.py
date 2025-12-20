"""
Generate leaderboard tables from aggregated metrics.
Supports multiple properties (formation_energy, stress_pressure, etc.) with property-specific metrics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

LOGGER = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Decimal places for metric formatting
METRIC_DECIMALS = {
    "mae": 3,  # MAE: 3 decimals (e.g., 0.045)
    "r_squared": 2,  # R²: 2 decimals (e.g., 0.98)
    "precision": 2,  # Precision: 2 decimals
    "recall": 2,  # Recall: 2 decimals
    "accuracy": 2,  # Accuracy: 2 decimals
}

DATASET_ORDER = [
    "convex_hull",
    "binary_alloys",
    "liquid_alloys",
    "random_alloys",
]

# Metrics for each property type
PROPERTY_METRICS = {
    "formation_energy": {
        "metric_order": [
            "mae",
            "r_squared",
            "n_structures",
            "precision",
            "recall",
            "accuracy",
            "true_positive_count",
            "false_positive_count",
            "false_negative_count",
            "true_negative_count",
        ],
        "minimal": [
            "mae",
            "r_squared",
            "n_structures",
            "precision",
            "recall",
            "accuracy",
        ],
    },
    "stress_pressure": {
        "metric_order": [
            "mae",
            "r_squared",
            "n_structures",
        ],
        "minimal": [
            "mae",
            "r_squared",
            "n_structures",
        ],
    },
    "potential_energy": {
        "metric_order": [
            "mae",
            "r_squared",
            "n_structures",
        ],
        "minimal": [
            "mae",
            "r_squared",
            "n_structures",
        ],
    },
}

METRIC_HEADERS = {
    "mae": "MAE",
    "r_squared": "R²",
    "n_structures": "N",
    "precision": "Precision",
    "recall": "Recall",
    "accuracy": "Accuracy",
    "true_positive_count": "TP",
    "false_positive_count": "FP",
    "false_negative_count": "FN",
    "true_negative_count": "TN",
}

# Human-readable property names
PROPERTY_NAMES = {
    "formation_energy": "Formation Energy",
    "stress_pressure": "Stress/Pressure",
    "potential_energy": "Potential Energy",
}


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================


def _load_metrics(filepath: Path) -> List[Dict[str, Any]]:
    """Load aggregated metrics from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def _get_available_properties(metrics: List[Dict]) -> List[str]:
    """Get list of available properties in metrics."""
    properties = set()
    for metric in metrics:
        if "property" in metric:
            properties.add(metric["property"])
    return sorted(list(properties))


def _filter_by_property(metrics: List[Dict], property_name: str) -> List[Dict]:
    """Filter metrics by property type."""
    filtered = [m for m in metrics if m.get("property") == property_name]
    if not filtered:
        raise ValueError(f"No metrics found for property: {property_name}")
    return filtered


def _group_by_dataset(metrics: List[Dict]) -> Dict[str, List[Dict]]:
    """Group metrics by dataset name."""
    groups = {}
    for metric in metrics:
        dataset = metric["dataset"]
        if dataset not in groups:
            groups[dataset] = []
        groups[dataset].append(metric)
    return groups


def _sort_by_mae(metrics: List[Dict]) -> List[Dict]:
    """Sort metrics by MAE ascending (lower is better)."""
    return sorted(metrics, key=lambda m: m["mae"])


# ============================================================================
# FORMATTING
# ============================================================================


def _format_metric_value(metric_name: str, value: float) -> str:
    """Format metric value based on type using configured decimal places."""
    if metric_name in METRIC_DECIMALS:
        decimals = METRIC_DECIMALS[metric_name]
        return f"{value:.{decimals}f}"
    elif metric_name in [
        "true_positive_count",
        "false_positive_count",
        "false_negative_count",
        "true_negative_count",
        "n_structures",
    ]:
        return str(int(value))
    else:
        return str(value)


def _create_markdown_table(
    metrics: List[Dict], dataset_name: str, property_name: str, show_all_metrics: bool
) -> str:
    """Create markdown table for a dataset."""
    # Get metrics for this property
    if property_name not in PROPERTY_METRICS:
        raise ValueError(f"Unknown property: {property_name}")

    property_config = PROPERTY_METRICS[property_name]
    metrics_to_show = (
        property_config["metric_order"]
        if show_all_metrics
        else property_config["minimal"]
    )

    # Build header with units for MAE
    headers = ["GNN"]
    for m in metrics_to_show:
        if m == "mae":
            # Add units based on property
            if property_name == "formation_energy":
                headers.append("MAE (eV/atom)")
            elif property_name == "stress_pressure":
                headers.append("MAE (eV/Å)")
            else:
                headers.append(METRIC_HEADERS[m])
        else:
            headers.append(METRIC_HEADERS[m])

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "|" + "|".join(["---" for _ in headers]) + "|"

    # Build rows
    rows = []
    sorted_metrics = _sort_by_mae(metrics)
    for metric in sorted_metrics:
        row_values = [metric["gnn_name"]]
        for metric_name in metrics_to_show:
            value = metric.get(metric_name)
            if value is not None:
                row_values.append(_format_metric_value(metric_name, value))
            else:
                row_values.append("N/A")
        rows.append("| " + " | ".join(row_values) + " |")

    # Combine all parts
    table = f"## Dataset: {dataset_name}\n\n"
    table += header_line + "\n"
    table += separator_line + "\n"
    table += "\n".join(rows) + "\n"

    return table


def _sort_datasets(dataset_names):
    """Sort datasets by predefined order, with unknown datasets at the end."""

    def sort_key(name):
        if name in DATASET_ORDER:
            return DATASET_ORDER.index(name)
        else:
            return len(DATASET_ORDER)  # Put unknowns at end

    return sorted(dataset_names, key=sort_key)


# ============================================================================
# MAIN GENERATION
# ============================================================================


def generate_leaderboard(
    metrics_file: Path,
    output_file: Path,
    property_name: str,
    show_all_metrics: bool = True,
):
    """Main function to generate leaderboard markdown for a specific property."""
    # Load metrics
    all_metrics = _load_metrics(metrics_file)

    # Filter by property
    property_metrics = _filter_by_property(all_metrics, property_name)

    # Extract threshold (only meaningful for formation_energy)
    threshold = property_metrics[0].get("stability_threshold")

    # Group by dataset
    dataset_groups = _group_by_dataset(property_metrics)

    # Build markdown content
    property_display_name = PROPERTY_NAMES.get(property_name, property_name)
    markdown_content = f"# GNN Leaderboard - {property_display_name}\n\n"

    if threshold is not None:
        markdown_content += f"**Stability Threshold:** {threshold} eV/atom\n\n"

    # Create table for each dataset
    for dataset_name in _sort_datasets(dataset_groups.keys()):
        dataset_metrics = dataset_groups[dataset_name]
        table = _create_markdown_table(
            dataset_metrics, dataset_name, property_name, show_all_metrics
        )
        markdown_content += table + "\n"

    # Save to file
    with open(output_file, "w") as f:
        f.write(markdown_content)

    LOGGER.info(f"Leaderboard saved to: {output_file}")
