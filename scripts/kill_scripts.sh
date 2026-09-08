#!/bin/bash

# HOW TO RUN: ./init_demo.sh node_list.txt
# This script assumes that the model will be run with all layers

# ============================
#        CONFIGURATION
# ============================

# Define the log directory and ensure it exists
LOG_DIR="$HOME/RESEARCH/rasp_compression/logs"
mkdir -p "$LOG_DIR"

# Function to log messages with timestamps
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/kill_scripts.log"
}

# Define the SSH password (Consider using SSH keys for enhanced security)
SSH_PASSWORD="${SSH_PASSWORD:-CHANGE_ME}"

# Starting port for layer allocation
START_PORT=12346

# Declare associative arrays
declare -A layer_port_map    # Maps layer number to port
declare -A layer_ip_map      # Maps layer number to IP
declare -A node_user_map     # Maps IP to node type (username)
declare -a entries           # Regular layer entries (layers 1-17)
declare -a layer0_entries    # Layer 0 entries

# Initialize max_layer
max_layer=0

# ============================
#      DEPENDENCY CHECK
# ============================

log "Starting setup script."
# sleep 1

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    log "ERROR: sshpass is not installed. Please install it before running this script."
    exit 1
fi
log "Dependency check passed: sshpass is installed."
# sleep 1

# ============================
#      INPUT VALIDATION
# ============================

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    log "ERROR: Incorrect number of arguments."
    log "Usage: $0 <node_list_file>"
    exit 1
fi

NODE_LIST_FILE="$1"

# Check if the node list file exists
if [ ! -f "$NODE_LIST_FILE" ]; then
    log "ERROR: Node list file '$NODE_LIST_FILE' does not exist."
    exit 1
fi

log "Input verification passed. Using node list file: $NODE_LIST_FILE."
# sleep 2

# ============================
#      PROCESS NODE LIST
# ============================

log "Processing node list from '$NODE_LIST_FILE'."
# sleep 2  # Wait for 2 seconds

while IFS='-' read -r node_type ip layers; do
    # Remove possible spaces around node_type
    node_type=$(echo "$node_type" | xargs)
    log "Reading node: Type='$node_type', IP='$ip', Layers='$layers'"
    # sleep 2  # Wait for 2 seconds
    node_user_map["$ip"]="$node_type"
done < "$NODE_LIST_FILE"

log "All nodes and layers have been read and processed."
# sleep 2  # Wait for 2 seconds

# ============================
#      CONFIGURATION FUNCTIONS
# ============================

# Function to kill a specific script if running
kill_script() {
    local ip="$1"
    local node_type="$2"
    local script="$3"
    local node_log="$4"

    # Find the process ID(s) of the script
    pids=$(sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no "$node_type@$ip" "pgrep -f '$script'")

    if [ -n "$pids" ]; then
        # Replace newlines with spaces
        pids_formatted=$(echo "$pids" | tr '\n' ' ')
        log "Found running $script with PID(s): $pids_formatted on Node '$node_type' ($ip)" | tee -a "$LOG_DIR/${node_log}"
        # Kill the process(es)
        sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no "$node_type@$ip" "kill $pids_formatted"
        log "Killed $script with PID(s): $pids_formatted on Node '$node_type' ($ip)" | tee -a "$LOG_DIR/${node_log}"
    else
        log "No running instance of $script found on Node '$node_type' ($ip)." | tee -a "$LOG_DIR/${node_log}"
    fi
}


# ============================
#    INITIAL SSH CONNECTION CHECK AND KILL RUNNING PROCESSES
# ============================

log "Performing initial SSH connection checks and killing any running 'resilientNode.py' scripts on all nodes."

for ip in "${!node_user_map[@]}"; do
    node_type="${node_user_map[$ip]}"
    node_log="${node_type}_${ip}.log"
    log "Checking SSH connection and killing processes on Node '$node_type' ($ip)" | tee -a "$LOG_DIR/${node_log}"

    # Try to SSH into the node
    sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$node_type@$ip" "echo 'SSH connection successful.'" &> /dev/null

    if [ "$?" -ne 0 ]; then
        log "ERROR: Unable to SSH into Node '$node_type' ($ip)." | tee -a "$LOG_DIR/${node_log}"
        continue
    else
        log "SSH connection successful to Node '$node_type' ($ip)." | tee -a "$LOG_DIR/${node_log}"
        # Kill 'resilientNode.py' processes
        kill_script "$ip" "$node_type" "resilientNode.py" "$node_log"
    fi
    sleep 1
done

log "All running processes killed."
