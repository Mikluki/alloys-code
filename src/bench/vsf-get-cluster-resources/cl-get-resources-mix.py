#!/usr/bin/env python3
"""
SLURM Cluster Capacity Audit Tool
Collects hardware specifications and partition limits for capacity planning.
Enhanced with usable resource tracking.
"""

import json
import re
import subprocess
import sys
from datetime import datetime


def run_command(cmd):
    """Execute shell command and return output."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command '{cmd}': {e}")
        return None


def parse_memory(mem_str):
    """Convert memory string to MB for consistent comparison."""
    if not mem_str or mem_str == "N/A":
        return 0

    mem_str = mem_str.upper()
    if "T" in mem_str:
        return int(float(mem_str.replace("T", "")) * 1024 * 1024)
    elif "G" in mem_str:
        return int(float(mem_str.replace("G", "")) * 1024)
    elif "M" in mem_str:
        return int(float(mem_str.replace("M", "")))
    else:
        return int(mem_str)


def parse_core_allocation(allocation_string):
    """Parse core allocation string: 'allocated/idle/other/total'."""
    try:
        parts = allocation_string.split("/")
        if len(parts) == 4:
            return {
                "allocated": int(parts[0]),
                "idle": int(parts[1]),
                "other": int(parts[2]),
                "total": int(parts[3]),
            }
    except (ValueError, AttributeError):
        pass
    return {"allocated": 0, "idle": 0, "other": 0, "total": 0}


def is_node_usable_mix_included(state):
    """Check if node is usable when including mix nodes."""
    return state in ["idle", "mix"]


def is_node_usable_mix_dropped(state):
    """Check if node is usable when dropping mix nodes."""
    return state == "idle"


def get_usable_cores_mix_included(state, total_cores):
    """Get usable cores when treating mix as fully available."""
    if is_node_usable_mix_included(state):
        return total_cores
    return 0


def get_usable_cores_mix_dropped(state, total_cores):
    """Get usable cores when dropping mix nodes entirely."""
    if is_node_usable_mix_dropped(state):
        return total_cores
    return 0


def calculate_usable_resources(node_details, include_mix=True):
    """Calculate usable resources with different mix node handling."""
    usable_data = {
        "total_nodes": 0,
        "total_cores": 0,
        "total_memory_gb": 0,
        "by_partition": {},
        "node_details": [],
    }

    for node in node_details:
        if include_mix:
            is_usable = is_node_usable_mix_included(node["state"])
            usable_cores = get_usable_cores_mix_included(node["state"], node["cores"])
        else:
            is_usable = is_node_usable_mix_dropped(node["state"])
            usable_cores = get_usable_cores_mix_dropped(node["state"], node["cores"])

        if is_usable:
            usable_data["total_nodes"] += 1
            usable_data["total_cores"] += usable_cores
            usable_data["total_memory_gb"] += node["memory_gb"]
            usable_data["node_details"].append(node)

            # Aggregate by partition
            for partition in node["partitions"]:
                if partition not in usable_data["by_partition"]:
                    usable_data["by_partition"][partition] = {
                        "nodes": 0,
                        "total_cores": 0,
                        "total_memory_gb": 0,
                        "node_types": {},
                    }

                usable_data["by_partition"][partition]["nodes"] += 1
                usable_data["by_partition"][partition]["total_cores"] += usable_cores
                usable_data["by_partition"][partition]["total_memory_gb"] += node[
                    "memory_gb"
                ]

                # Track node types
                node_type = f"{node['cores']}c_{round(node['memory_mb']/1024)}GB"
                if (
                    node_type
                    not in usable_data["by_partition"][partition]["node_types"]
                ):
                    usable_data["by_partition"][partition]["node_types"][node_type] = 0
                usable_data["by_partition"][partition]["node_types"][node_type] += 1

    return usable_data


def get_cluster_specs(cluster_name):
    """Get cluster hardware specifications using sinfo."""
    print(f"Gathering cluster specifications for {cluster_name}...")

    # Get detailed node information
    sinfo_cmd = "sinfo -N -o '%N %c %m %P %t %C' --noheader"
    sinfo_output = run_command(sinfo_cmd)

    if not sinfo_output:
        return None

    cluster_data = {
        "cluster_name": cluster_name,
        "audit_timestamp": datetime.now().isoformat(),
        "total_nodes": 0,
        "total_cores": 0,
        "partitions": {},
        "node_details": [],
    }

    # Parse node information
    for line in sinfo_output.split("\n"):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) < 6:
            continue

        node_name = parts[0]
        cores = int(parts[1]) if parts[1].isdigit() else 0
        memory_mb = parse_memory(parts[2])
        partitions = parts[3].split(",")
        state = parts[4]
        core_info = parts[5]  # Format: allocated/idle/other/total

        node_info = {
            "name": node_name,
            "cores": cores,
            "memory_mb": memory_mb,
            "memory_gb": round(memory_mb / 1024, 1),
            "partitions": partitions,
            "state": state,
            "core_allocation": core_info,
        }

        cluster_data["node_details"].append(node_info)
        cluster_data["total_nodes"] += 1
        cluster_data["total_cores"] += cores

        # Aggregate by partition
        for partition in partitions:
            if partition not in cluster_data["partitions"]:
                cluster_data["partitions"][partition] = {
                    "nodes": 0,
                    "total_cores": 0,
                    "total_memory_gb": 0,
                    "node_types": {},
                }

            cluster_data["partitions"][partition]["nodes"] += 1
            cluster_data["partitions"][partition]["total_cores"] += cores
            cluster_data["partitions"][partition]["total_memory_gb"] += round(
                memory_mb / 1024, 1
            )

            # Track node types within partition
            node_type = f"{cores}c_{round(memory_mb/1024)}GB"
            if node_type not in cluster_data["partitions"][partition]["node_types"]:
                cluster_data["partitions"][partition]["node_types"][node_type] = 0
            cluster_data["partitions"][partition]["node_types"][node_type] += 1

    return cluster_data


def get_partition_limits(cluster_name):
    """Get partition policies and limits using scontrol."""
    print(f"Gathering partition limits for {cluster_name}...")

    # Get partition information
    scontrol_cmd = "scontrol show partition --oneliner"
    scontrol_output = run_command(scontrol_cmd)

    if not scontrol_output:
        return {}

    partition_limits = {}

    for line in scontrol_output.split("\n"):
        if not line.strip():
            continue

        # Parse key=value pairs
        partition_info = {}
        for item in line.split():
            if "=" in item:
                key, value = item.split("=", 1)
                partition_info[key] = value

        if "PartitionName" not in partition_info:
            continue

        partition_name = partition_info["PartitionName"]

        partition_limits[partition_name] = {
            "max_time": partition_info.get("MaxTime", "UNLIMITED"),
            "default_time": partition_info.get("DefaultTime", "NONE"),
            "max_nodes": partition_info.get("MaxNodes", "UNLIMITED"),
            "max_cpus_per_user": partition_info.get("MaxCPUsPerUser", "UNLIMITED"),
            "max_jobs_per_user": partition_info.get("MaxJobsPerUser", "UNLIMITED"),
            "state": partition_info.get("State", "UNKNOWN"),
            "priority_tier": partition_info.get("PriorityTier", "1"),
            "preempt_mode": partition_info.get("PreemptMode", "OFF"),
            "allowed_accounts": partition_info.get("AllowAccounts", "ALL"),
        }

    return partition_limits


def export_cluster_summary(cluster_data, partition_limits, cluster_name):
    """Combine and export cluster information to JSON."""

    # Merge partition limits into cluster data
    for partition_name, partition_data in cluster_data["partitions"].items():
        if partition_name in partition_limits:
            partition_data["limits"] = partition_limits[partition_name]
        else:
            partition_data["limits"] = {"status": "limits_not_found"}

    # Calculate usable resources with different strategies
    cluster_data["usable_resources_mix_included"] = calculate_usable_resources(
        cluster_data["node_details"], include_mix=True
    )
    cluster_data["usable_resources_mix_dropped"] = calculate_usable_resources(
        cluster_data["node_details"], include_mix=False
    )

    # Add summary statistics
    cluster_data["summary"] = {
        "total_partitions": len(cluster_data["partitions"]),
        "average_cores_per_node": (
            round(cluster_data["total_cores"] / cluster_data["total_nodes"], 1)
            if cluster_data["total_nodes"] > 0
            else 0
        ),
        "total_memory_gb": sum(
            node["memory_gb"] for node in cluster_data["node_details"]
        ),
        "largest_partition": (
            max(cluster_data["partitions"].items(), key=lambda x: x[1]["total_cores"])[
                0
            ]
            if cluster_data["partitions"]
            else None
        ),
        "usable_efficiency_mix_included": (
            round(
                cluster_data["usable_resources_mix_included"]["total_cores"]
                / cluster_data["total_cores"]
                * 100,
                1,
            )
            if cluster_data["total_cores"] > 0
            else 0
        ),
        "usable_efficiency_mix_dropped": (
            round(
                cluster_data["usable_resources_mix_dropped"]["total_cores"]
                / cluster_data["total_cores"]
                * 100,
                1,
            )
            if cluster_data["total_cores"] > 0
            else 0
        ),
    }

    # Export to JSON file
    filename = f"{cluster_name}_capacity.json"
    with open(filename, "w") as f:
        json.dump(cluster_data, f, indent=2)

    print(f"\nCluster capacity data exported to: {filename}")
    print(f"Summary:")
    print(f"  Total nodes: {cluster_data['total_nodes']}")
    print(f"  Total cores: {cluster_data['total_cores']}")
    print(f"  Total memory: {cluster_data['summary']['total_memory_gb']:.1f} GB")
    print(f"  Partitions: {cluster_data['summary']['total_partitions']}")
    print(f"\nUsable Resources:")
    print(
        f"  Mix included - Nodes: {cluster_data['usable_resources_mix_included']['total_nodes']}, "
        f"Cores: {cluster_data['usable_resources_mix_included']['total_cores']} "
        f"({cluster_data['summary']['usable_efficiency_mix_included']}%)"
    )
    print(
        f"  Mix dropped  - Nodes: {cluster_data['usable_resources_mix_dropped']['total_nodes']}, "
        f"Cores: {cluster_data['usable_resources_mix_dropped']['total_cores']} "
        f"({cluster_data['summary']['usable_efficiency_mix_dropped']}%)"
    )

    return filename


def main():
    if len(sys.argv) != 2:
        print("Usage: python cluster_audit.py <cluster_name>")
        sys.exit(1)

    cluster_name = sys.argv[1]
    print(f"Starting cluster audit for: {cluster_name}")

    # Get cluster specifications
    cluster_data = get_cluster_specs(cluster_name)
    if not cluster_data:
        print("Failed to gather cluster specifications")
        sys.exit(1)

    # Get partition limits
    partition_limits = get_partition_limits(cluster_name)

    # Export combined data
    export_cluster_summary(cluster_data, partition_limits, cluster_name)

    print(f"\nCluster audit completed for {cluster_name}")


if __name__ == "__main__":
    main()
