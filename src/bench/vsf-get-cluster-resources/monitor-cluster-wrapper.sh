#!/bin/bash

# SLURM Cluster Monitoring Wrapper
# Runs load monitoring every hour for specified days to build usage patterns

set -euo pipefail

# Default values
CLUSTER_NAME=""
DAYS=7
INTERVAL_HOURS=1
PYTHON_SCRIPT="cl-load-monitor.py"
LOG_DIR="monitoring_logs"
DATA_DIR="monitoring_data"
PID_FILE=""
ACTION="start"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <cluster_name> [options]"
    echo ""
    echo "Options:"
    echo "  -d, --days DAYS           Number of days to monitor (default: 7)"
    echo "  -i, --interval HOURS      Hours between monitoring runs (default: 1)"
    echo "  -a, --action ACTION       Action: start|stop|status|analyze (default: start)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 staer                 # Monitor 'staer' for 7 days, every hour"
    echo "  $0 staer -d 3 -i 2       # Monitor 'staer' for 3 days, every 2 hours"
    echo "  $0 staer -a stop         # Stop monitoring 'staer'"
    echo "  $0 staer -a status       # Check monitoring status for 'staer'"
    echo "  $0 staer -a analyze      # Analyze collected data for 'staer'"
    exit 1
}

log_message() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${GREEN}[INFO]${NC} ${timestamp}: $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} ${timestamp}: $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} ${timestamp}: $message" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} ${timestamp}: $message" ;;
    esac
    
    # Also log to file
    echo "[$level] ${timestamp}: $message" >> "${LOG_DIR}/${CLUSTER_NAME}_monitor.log"
}

setup_directories() {
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "${DATA_DIR}/${CLUSTER_NAME}"
    
    log_message "INFO" "Created monitoring directories"
}

check_dependencies() {
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        log_message "ERROR" "Python script '$PYTHON_SCRIPT' not found in current directory"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        log_message "ERROR" "python3 not found"
        exit 1
    fi
    
    if ! command -v squeue &> /dev/null; then
        log_message "ERROR" "squeue command not found - SLURM not available"
        exit 1
    fi
    
    log_message "INFO" "All dependencies found"
}

start_monitoring() {
    if is_monitoring_active; then
        log_message "WARN" "Monitoring for cluster '$CLUSTER_NAME' is already running (PID: $(cat $PID_FILE))"
        exit 1
    fi
    
    local total_runs=$((DAYS * 24 / INTERVAL_HOURS))
    local end_time=$(date -d "+${DAYS} days" '+%Y-%m-%d %H:%M:%S')
    
    log_message "INFO" "Starting monitoring for cluster '$CLUSTER_NAME'"
    log_message "INFO" "Duration: $DAYS days, Interval: $INTERVAL_HOURS hours"
    log_message "INFO" "Total expected runs: $total_runs"
    log_message "INFO" "End time: $end_time"
    
    # Start background monitoring process
    (monitor_loop) &
    local monitor_pid=$!
    echo $monitor_pid > "$PID_FILE"
    
    log_message "INFO" "Background monitoring started with PID: $monitor_pid"
    log_message "INFO" "Use '$0 $CLUSTER_NAME -a status' to check progress"
    log_message "INFO" "Use '$0 $CLUSTER_NAME -a stop' to stop monitoring"
}

monitor_loop() {
    local start_time=$(date +%s)
    local end_time=$((start_time + DAYS * 24 * 3600))
    local run_count=0
    
    while [[ $(date +%s) -lt $end_time ]]; do
        run_count=$((run_count + 1))
        local current_time=$(date '+%Y-%m-%d %H:%M:%S')
        
        log_message "INFO" "Starting monitoring run #$run_count at $current_time"
        
        # Run the Python monitoring script
        local output_file="${DATA_DIR}/${CLUSTER_NAME}/run_${run_count}_$(date +%Y%m%d_%H%M%S).json"
        
        if python3 "$PYTHON_SCRIPT" "$CLUSTER_NAME" > "${LOG_DIR}/${CLUSTER_NAME}_run_${run_count}.log" 2>&1; then
            # Move the generated JSON file to our organized structure
            local latest_json=$(ls -t ${CLUSTER_NAME}_current_load_*.json 2>/dev/null | head -1)
            if [[ -n "$latest_json" ]]; then
                mv "$latest_json" "$output_file"
                log_message "INFO" "Run #$run_count completed successfully. Data saved to: $output_file"
            else
                log_message "WARN" "Run #$run_count completed but no output file found"
            fi
        else
            log_message "ERROR" "Run #$run_count failed. Check log: ${LOG_DIR}/${CLUSTER_NAME}_run_${run_count}.log"
        fi
        
        # Calculate remaining time and runs
        local current_timestamp=$(date +%s)
        local remaining_time=$((end_time - current_timestamp))
        local remaining_runs=$(((end_time - current_timestamp) / (INTERVAL_HOURS * 3600)))
        
        if [[ $remaining_time -gt 0 ]]; then
            log_message "INFO" "Waiting $INTERVAL_HOURS hours until next run. Remaining: $remaining_runs runs"
            sleep $((INTERVAL_HOURS * 3600))
        fi
    done
    
    log_message "INFO" "Monitoring completed after $run_count runs"
    rm -f "$PID_FILE"
}

stop_monitoring() {
    if ! is_monitoring_active; then
        log_message "WARN" "No monitoring process found for cluster '$CLUSTER_NAME'"
        exit 1
    fi
    
    local pid=$(cat "$PID_FILE")
    log_message "INFO" "Stopping monitoring process (PID: $pid)"
    
    if kill "$pid" 2>/dev/null; then
        rm -f "$PID_FILE"
        log_message "INFO" "Monitoring stopped successfully"
    else
        log_message "ERROR" "Failed to stop monitoring process. Process may have already ended."
        rm -f "$PID_FILE"
    fi
}

monitoring_status() {
    if is_monitoring_active; then
        local pid=$(cat "$PID_FILE")
        local data_files=$(ls "${DATA_DIR}/${CLUSTER_NAME}/"*.json 2>/dev/null | wc -l)
        
        log_message "INFO" "Monitoring Status for '$CLUSTER_NAME':"
        echo "  Status: RUNNING (PID: $pid)"
        echo "  Data files collected: $data_files"
        echo "  Log file: ${LOG_DIR}/${CLUSTER_NAME}_monitor.log"
        echo "  Data directory: ${DATA_DIR}/${CLUSTER_NAME}/"
        
        if [[ $data_files -gt 0 ]]; then
            local latest_file=$(ls -t "${DATA_DIR}/${CLUSTER_NAME}/"*.json | head -1)
            local latest_time=$(stat -c %y "$latest_file" | cut -d'.' -f1)
            echo "  Latest data: $latest_time"
        fi
    else
        echo "Monitoring Status for '$CLUSTER_NAME': NOT RUNNING"
        
        local data_files=$(ls "${DATA_DIR}/${CLUSTER_NAME}/"*.json 2>/dev/null | wc -l)
        if [[ $data_files -gt 0 ]]; then
            echo "  Previous data files: $data_files"
            echo "  Use '$0 $CLUSTER_NAME -a analyze' to analyze collected data"
        fi
    fi
}

is_monitoring_active() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat $PID_FILE)" 2>/dev/null
}

analyze_data() {
    local data_files=("${DATA_DIR}/${CLUSTER_NAME}/"*.json)
    
    if [[ ! -f "${data_files[0]}" ]]; then
        log_message "ERROR" "No data files found for cluster '$CLUSTER_NAME'"
        exit 1
    fi
    
    local file_count=${#data_files[@]}
    log_message "INFO" "Analyzing $file_count data files for cluster '$CLUSTER_NAME'"
    
    # Create analysis summary
    local analysis_file="${DATA_DIR}/${CLUSTER_NAME}/analysis_summary_$(date +%Y%m%d_%H%M%S).json"
    
    python3 -c "
import json
import sys
from glob import glob
from datetime import datetime
from collections import defaultdict
import statistics

data_files = glob('${DATA_DIR}/${CLUSTER_NAME}/*.json')
data_files = [f for f in data_files if 'analysis_summary' not in f]

if not data_files:
    print('No data files to analyze')
    sys.exit(1)

hourly_utilization = defaultdict(list)
daily_utilization = defaultdict(list)
partition_stats = defaultdict(lambda: defaultdict(list))

for file_path in sorted(data_files):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        hour = timestamp.hour
        day = timestamp.strftime('%A').lower()
        
        # Extract utilization data
        if 'node_utilization' in data and 'partition_utilization' in data['node_utilization']:
            for partition, util_data in data['node_utilization']['partition_utilization'].items():
                util_pct = util_data.get('utilization_pct', 0)
                hourly_utilization[hour].append(util_pct)
                daily_utilization[day].append(util_pct)
                partition_stats[partition]['utilization'].append(util_pct)
                partition_stats[partition]['idle_cores'].append(util_data.get('idle_cores', 0))
                
    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        continue

# Calculate summary statistics
analysis = {
    'cluster_name': '${CLUSTER_NAME}',
    'analysis_timestamp': datetime.now().isoformat(),
    'files_analyzed': len(data_files),
    'hourly_patterns': {},
    'daily_patterns': {},
    'partition_analysis': {},
    'recommendations': []
}

# Hourly patterns
for hour in range(24):
    if hour in hourly_utilization:
        utils = hourly_utilization[hour]
        analysis['hourly_patterns'][f'{hour:02d}:00'] = {
            'avg_utilization': round(statistics.mean(utils), 1),
            'min_utilization': round(min(utils), 1),
            'max_utilization': round(max(utils), 1),
            'samples': len(utils)
        }

# Daily patterns
for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
    if day in daily_utilization:
        utils = daily_utilization[day]
        analysis['daily_patterns'][day] = {
            'avg_utilization': round(statistics.mean(utils), 1),
            'min_utilization': round(min(utils), 1),
            'max_utilization': round(max(utils), 1),
            'samples': len(utils)
        }

# Partition analysis
for partition, stats in partition_stats.items():
    if stats['utilization']:
        analysis['partition_analysis'][partition] = {
            'avg_utilization': round(statistics.mean(stats['utilization']), 1),
            'avg_idle_cores': round(statistics.mean(stats['idle_cores']), 1),
            'max_idle_cores': max(stats['idle_cores']),
            'samples': len(stats['utilization'])
        }

# Generate recommendations
best_hours = sorted(analysis['hourly_patterns'].items(), key=lambda x: x[1]['avg_utilization'])[:3]
worst_hours = sorted(analysis['hourly_patterns'].items(), key=lambda x: x[1]['avg_utilization'])[-3:]

analysis['recommendations'] = [
    f'Best hours for submission: {', '.join([h[0] for h in best_hours])}',
    f'Avoid these hours: {', '.join([h[0] for h in worst_hours])}',
]

if analysis['daily_patterns']:
    best_days = sorted(analysis['daily_patterns'].items(), key=lambda x: x[1]['avg_utilization'])[:2]
    analysis['recommendations'].append(f'Best days: {', '.join([d[0].title() for d in best_days])}')

with open('${analysis_file}', 'w') as f:
    json.dump(analysis, f, indent=2)

print(f'Analysis complete. Results saved to: ${analysis_file}')
print(f'Files analyzed: {len(data_files)}')
print('Summary:')
for rec in analysis['recommendations']:
    print(f'  - {rec}')
"
    
    log_message "INFO" "Analysis completed. Results saved to: $analysis_file"
}

# Parse command line arguments
if [[ $# -eq 0 ]]; then
    usage
fi

CLUSTER_NAME=$1
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--days)
            DAYS="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL_HOURS="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate inputs
if [[ ! "$DAYS" =~ ^[0-9]+$ ]] || [[ $DAYS -lt 1 ]]; then
    echo "Error: Days must be a positive integer"
    exit 1
fi

if [[ ! "$INTERVAL_HOURS" =~ ^[0-9]+$ ]] || [[ $INTERVAL_HOURS -lt 1 ]]; then
    echo "Error: Interval must be a positive integer"
    exit 1
fi

# Set PID file path
PID_FILE="${LOG_DIR}/${CLUSTER_NAME}_monitor.pid"

# Create directories
setup_directories

# Execute action
case $ACTION in
    "start")
        check_dependencies
        start_monitoring
        ;;
    "stop")
        stop_monitoring
        ;;
    "status")
        monitoring_status
        ;;
    "analyze")
        analyze_data
        ;;
    *)
        echo "Error: Unknown action '$ACTION'"
        echo "Valid actions: start, stop, status, analyze"
        exit 1
        ;;
esac
