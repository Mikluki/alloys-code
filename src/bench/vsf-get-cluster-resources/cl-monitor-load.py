#!/usr/bin/env python3
"""
SLURM Current Load Monitor
Since sacct historical data is unavailable, this script focuses on current queue analysis
and can be run periodically to build usage patterns over time.
"""

import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta


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


def get_detailed_queue_status(exclude_users):
    """Get detailed current queue information."""
    print("Analyzing current queue status...")

    # Enhanced squeue command with more details
    squeue_cmd = "squeue -o '%i|%P|%u|%T|%C|%M|%l|%D|%R' --noheader"
    squeue_output = run_command(squeue_cmd)

    if not squeue_output:
        return {}

    current_status = {
        "timestamp": datetime.now().isoformat(),
        "running_jobs": [],
        "pending_jobs": [],
        "partition_summary": defaultdict(
            lambda: {
                "running_jobs": 0,
                "pending_jobs": 0,
                "cores_running": 0,
                "cores_pending": 0,
                "users": set(),
            }
        ),
        "user_summary": defaultdict(
            lambda: {
                "running_jobs": 0,
                "pending_jobs": 0,
                "cores_running": 0,
                "cores_pending": 0,
            }
        ),
        "queue_reasons": defaultdict(int),
    }

    for line in squeue_output.split("\n"):
        if not line.strip():
            continue

        parts = line.split("|")
        if len(parts) < 9:
            continue

        job_id, partition, user, state, cores, runtime, timelimit, nodes, reason = parts

        # Skip jobs from excluded users
        if any(excl_user in user.lower() for excl_user in exclude_users):
            continue

        cores_num = int(cores) if cores.isdigit() else 0
        nodes_num = int(nodes) if nodes.isdigit() else 0

        job_info = {
            "job_id": job_id,
            "partition": partition,
            "user": user,
            "cores": cores_num,
            "nodes": nodes_num,
            "runtime": runtime,
            "timelimit": timelimit,
            "reason": reason,
        }

        # Categorize by state
        if state in ["RUNNING", "R"]:
            current_status["running_jobs"].append(job_info)
            current_status["partition_summary"][partition]["running_jobs"] += 1
            current_status["partition_summary"][partition]["cores_running"] += cores_num
            current_status["partition_summary"][partition]["users"].add(user)
            current_status["user_summary"][user]["running_jobs"] += 1
            current_status["user_summary"][user]["cores_running"] += cores_num

        elif state in ["PENDING", "PD"]:
            current_status["pending_jobs"].append(job_info)
            current_status["partition_summary"][partition]["pending_jobs"] += 1
            current_status["partition_summary"][partition]["cores_pending"] += cores_num
            current_status["partition_summary"][partition]["users"].add(user)
            current_status["user_summary"][user]["pending_jobs"] += 1
            current_status["user_summary"][user]["cores_pending"] += cores_num
            current_status["queue_reasons"][reason] += 1

    # Convert sets to lists for JSON serialization
    for partition in current_status["partition_summary"]:
        current_status["partition_summary"][partition]["users"] = list(
            current_status["partition_summary"][partition]["users"]
        )
        current_status["partition_summary"][partition]["unique_users"] = len(
            current_status["partition_summary"][partition]["users"]
        )

    # Convert defaultdicts to regular dicts
    current_status["partition_summary"] = dict(current_status["partition_summary"])
    current_status["user_summary"] = dict(current_status["user_summary"])
    current_status["queue_reasons"] = dict(current_status["queue_reasons"])

    return current_status


def analyze_node_utilization():
    """Analyze current node utilization using sinfo."""
    print("Analyzing node utilization...")

    sinfo_cmd = "sinfo -N -o '%N|%c|%C|%P|%t' --noheader"
    sinfo_output = run_command(sinfo_cmd)

    if not sinfo_output:
        return {}

    node_analysis = {
        "total_nodes": 0,
        "nodes_by_state": defaultdict(int),
        "partition_utilization": defaultdict(
            lambda: {
                "total_cores": 0,
                "allocated_cores": 0,
                "idle_cores": 0,
                "utilization_pct": 0,
            }
        ),
    }

    for line in sinfo_output.split("\n"):
        if not line.strip():
            continue

        parts = line.split("|")
        if len(parts) < 5:
            continue

        node_name, total_cores, core_info, partitions, state = parts

        # Skip head node (typically node07 or similar)
        if "node07" in node_name.lower() or "head" in node_name.lower():
            continue

        node_analysis["total_nodes"] += 1
        node_analysis["nodes_by_state"][state] += 1

        # Parse core allocation: allocated/idle/other/total
        if "/" in core_info:
            core_parts = core_info.split("/")
            if len(core_parts) >= 4:
                allocated = int(core_parts[0]) if core_parts[0].isdigit() else 0
                idle = int(core_parts[1]) if core_parts[1].isdigit() else 0
                total = int(core_parts[3]) if core_parts[3].isdigit() else 0

                for partition in partitions.split(","):
                    node_analysis["partition_utilization"][partition][
                        "total_cores"
                    ] += total
                    node_analysis["partition_utilization"][partition][
                        "allocated_cores"
                    ] += allocated
                    node_analysis["partition_utilization"][partition][
                        "idle_cores"
                    ] += idle

    # Calculate utilization percentages
    for partition in node_analysis["partition_utilization"]:
        total = node_analysis["partition_utilization"][partition]["total_cores"]
        allocated = node_analysis["partition_utilization"][partition]["allocated_cores"]
        if total > 0:
            node_analysis["partition_utilization"][partition]["utilization_pct"] = (
                round((allocated / total) * 100, 1)
            )

    # Convert defaultdicts to regular dicts
    node_analysis["nodes_by_state"] = dict(node_analysis["nodes_by_state"])
    node_analysis["partition_utilization"] = dict(
        node_analysis["partition_utilization"]
    )

    return node_analysis


def generate_immediate_recommendations(queue_status, node_analysis):
    """Generate recommendations based on current state."""
    recommendations = {
        "immediate_opportunities": [],
        "partition_recommendations": {},
        "wait_time_estimates": {},
        "optimal_job_sizes": {},
    }

    # Analyze immediate opportunities
    total_idle_cores = sum(
        partition_data["idle_cores"]
        for partition_data in node_analysis["partition_utilization"].values()
    )

    if total_idle_cores > 32:
        recommendations["immediate_opportunities"].append(
            f"Good opportunity: {total_idle_cores} idle cores available across cluster"
        )
    elif total_idle_cores > 0:
        recommendations["immediate_opportunities"].append(
            f"Limited opportunity: {total_idle_cores} idle cores available"
        )
    else:
        recommendations["immediate_opportunities"].append(
            "No idle cores currently available - consider waiting"
        )

    # Partition-specific recommendations
    for partition, util_data in node_analysis["partition_utilization"].items():
        utilization_pct = util_data["utilization_pct"]
        idle_cores = util_data["idle_cores"]

        if utilization_pct < 50 and idle_cores > 16:
            recommendations["partition_recommendations"][
                partition
            ] = "Good availability - recommended"
        elif utilization_pct < 80 and idle_cores > 0:
            recommendations["partition_recommendations"][
                partition
            ] = "Moderate availability - consider small jobs"
        else:
            recommendations["partition_recommendations"][
                partition
            ] = "High utilization - avoid unless urgent"

    # Estimate wait times based on queue
    for partition, queue_data in queue_status["partition_summary"].items():
        pending_jobs = queue_data["pending_jobs"]
        if pending_jobs == 0:
            recommendations["wait_time_estimates"][
                partition
            ] = "No queue - immediate execution likely"
        elif pending_jobs < 5:
            recommendations["wait_time_estimates"][
                partition
            ] = "Short queue - moderate wait expected"
        else:
            recommendations["wait_time_estimates"][
                partition
            ] = f"Long queue ({pending_jobs} jobs) - significant wait expected"

    return recommendations


def export_current_analysis(cluster_name, queue_status, node_analysis, recommendations):
    """Export current load analysis."""

    analysis_data = {
        "cluster_name": cluster_name,
        "analysis_type": "current_load_snapshot",
        "timestamp": datetime.now().isoformat(),
        "queue_status": queue_status,
        "node_utilization": node_analysis,
        "recommendations": recommendations,
        "summary": {
            "total_running_jobs": len(queue_status["running_jobs"]),
            "total_pending_jobs": len(queue_status["pending_jobs"]),
            "total_active_users": len(queue_status["user_summary"]),
            "cluster_utilization_pct": round(
                (
                    sum(
                        p["utilization_pct"]
                        for p in node_analysis["partition_utilization"].values()
                    )
                    / len(node_analysis["partition_utilization"])
                    if node_analysis["partition_utilization"]
                    else 0
                ),
                1,
            ),
            "immediate_submission_recommended": any(
                "Good" in rec
                for rec in recommendations["partition_recommendations"].values()
            ),
        },
    }

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{cluster_name}_current_load_{timestamp_str}.json"

    with open(filename, "w") as f:
        json.dump(analysis_data, f, indent=2, default=str)

    print(f"\nCurrent load analysis exported to: {filename}")
    print(f"\nCurrent Status Summary:")
    print(f"  Running jobs: {analysis_data['summary']['total_running_jobs']}")
    print(f"  Pending jobs: {analysis_data['summary']['total_pending_jobs']}")
    print(f"  Active users: {analysis_data['summary']['total_active_users']}")
    print(
        f"  Cluster utilization: {analysis_data['summary']['cluster_utilization_pct']}%"
    )
    print(
        f"  Immediate submission recommended: {analysis_data['summary']['immediate_submission_recommended']}"
    )

    # Print partition status
    print(f"\nPartition Status:")
    for partition, data in node_analysis["partition_utilization"].items():
        print(
            f"  {partition}: {data['utilization_pct']}% utilized ({data['idle_cores']} idle cores)"
        )

    return filename


def main():
    if len(sys.argv) != 2:
        print("Usage: python current_load_monitor.py <cluster_name>")
        print("Example: python current_load_monitor.py staer")
        sys.exit(1)

    cluster_name = sys.argv[1]
    exclude_users = ["lukianov", "zhilyaev"]

    print(f"Starting current load analysis for cluster: {cluster_name}")
    print(f"Excluding users containing: {', '.join(exclude_users)}")
    print(
        "Note: This analysis focuses on current state since historical data is unavailable"
    )

    # Get current queue status
    queue_status = get_detailed_queue_status(exclude_users)

    # Get node utilization
    node_analysis = analyze_node_utilization()

    # Generate recommendations
    recommendations = generate_immediate_recommendations(queue_status, node_analysis)

    # Export analysis
    export_current_analysis(cluster_name, queue_status, node_analysis, recommendations)

    print(f"\nCurrent load monitoring completed for {cluster_name}")
    print("Run this script periodically to build usage patterns over time")


if __name__ == "__main__":
    main()
