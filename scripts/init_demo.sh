#!/bin/bash

# HOW TO RUN: ./init_demo.sh node_list.txt
# This script assumes that the model will be run with all layers

# ============================
#        CONFIGURATION
# ============================

# ============================
# Example configurations:
# ============================

# TopK 50%:
#   COMPRESSION_METHOD="topk"
#   COMPRESSION_RATIO="0.5"

# INT8 quantization:
#   COMPRESSION_METHOD="quantization"
#   COMPRESSION_RATIO="0.25"

# LLMInt8 (1% outliers FP16, rest INT8):
#   COMPRESSION_METHOD="llmint8"
#   COMPRESSION_RATIO="0.25"
#   LLMINT8_OUTLIER_RATIO="0.01"
#   LLMINT8_OUTLIER_PREC="fp16"
#   LLMINT8_REGULAR_PREC="int8"

# LLMInt8 (5% outliers FP16, rest INT4):
#   COMPRESSION_METHOD="llmint8"
#   COMPRESSION_RATIO="0.125"
#   LLMINT8_OUTLIER_RATIO="0.05"
#   LLMINT8_OUTLIER_PREC="fp16"
#   LLMINT8_REGULAR_PREC="int4"

# No compression:
#   COMPRESSION_METHOD="none"
#   COMPRESSION_RATIO="1.0"

COMPRESSION_METHOD="none"      # "topk" | "quantization" | "llmint8" | "none"

# Start ratio, then optimized at ControlDirective (or for competitors, set to a fixed value)
COMPRESSION_RATIO="1.0"        # eta: 0.5=TopK50%, 0.25=INT8, 0.125=INT4, 0.0625=INT2, etc.

# LLMInt8-specific params (only used when COMPRESSION_METHOD="llmint8")
LLMINT8_OUTLIER_RATIO="0.05"       # fraction of outlier elements (e.g. 0.01 = top 1%)
LLMINT8_OUTLIER_PREC="fp16"        # outlier precision: fp16 | int8
LLMINT8_REGULAR_PREC="int8"        # regular precision: fp16 | int8 | int4 | int2

CONTROLLER_HOST=""                 # auto-derived from the node list (Layer 0 / UE IP) below
CONTROLLER_PORT="9999"             # UDP port for metrics
CONTROL_BASE_PORT="10000"          # directive port = base + layer_id

# Define the log directory and ensure it exists
LOG_DIR="$HOME/RESEARCH/rasp_compression/logs"
mkdir -p "$LOG_DIR"

# Function to log messages with timestamps
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_DIR/setup.log"
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
sleep 1

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    log "ERROR: sshpass is not installed. Please install it before running this script."
    exit 1
fi
log "Dependency check passed: sshpass is installed."
sleep 1

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
sleep 1

# ============================
#      PROCESS NODE LIST
# ============================

log "Processing node list from '$NODE_LIST_FILE'."
sleep 1

while IFS='-' read -r node_type ip layers; do
    # Remove possible spaces around node_type
    node_type=$(echo "$node_type" | xargs)
    log "Reading node: Type='$node_type', IP='$ip', Layers='$layers'"
    sleep 1

    # Split layers by comma
    IFS=',' read -ra layer_array <<< "$layers"
    for layer in "${layer_array[@]}"; do
        # Trim any whitespace
        layer=$(echo "$layer" | xargs)
        # Update max_layer
        if [ "$layer" -gt "$max_layer" ]; then
            max_layer=$layer
        fi
        if [ "$layer" -eq 0 ]; then
            # Separate Layer 0 entries to process them last
            layer0_entries+=("$ip-$layer")
            log "Processed Layer $layer for IP $ip (will be processed last)."
        else
            entries+=("$ip-$layer")
            log "Processed Layer $layer for IP $ip."
            # Map layer to IP
            layer_ip_map["$layer"]="$ip"
        fi
        # Map IP to username (assuming node_type is the username)
        node_user_map["$ip"]="$node_type"
    done
done < "$NODE_LIST_FILE"

log "All nodes and layers have been read and processed."
sleep 2  # Wait for 2 seconds

# Compute total number of layers
N=$((max_layer + 1))
log "Total number of layers (N) is: $N"
sleep 2  # Wait for 2 seconds

# ============================
#        SORT ENTRIES
# ============================

# Combine sorted entries: Layers 1-17 first, then Layer 0
combined_entries=("${entries[@]}" "${layer0_entries[@]}")

log "All entries have been combined with Layers 1-17 first and Layer 0 last."
sleep 2  # Wait for 2 seconds

# ============================
#      ASSIGN PORTS TO LAYERS
# ============================

# Assign a unique port to each layer, starting from START_PORT
for entry in "${combined_entries[@]}"; do
    ip=$(echo "$entry" | cut -d'-' -f1)
    layer=$(echo "$entry" | cut -d'-' -f2)

    # Assign port only if not already assigned
    if [ -z "${layer_port_map[$layer]}" ]; then
        port=$((START_PORT + layer))
        layer_port_map["$layer"]=$port
        log "Assigned Port $port to Layer $layer."
    else
        port=${layer_port_map[$layer]}
        log "Layer $layer already assigned to Port $port."
    fi
    # Map layer to IP
    layer_ip_map["$layer"]="$ip"
done
sleep 2

# Log the layer to port mapping in sorted order
log "Layer to Port Mapping:"
for layer in $(echo "${!layer_port_map[@]}" | tr ' ' '\n' | sort -n); do
    log "Layer $layer → Port ${layer_port_map[$layer]}"
done

# ============================
#      VERIFY Layer 0 Exists
# ============================

if [ -z "${layer_ip_map[0]}" ]; then
    log "ERROR: No Layer 0 (sid) node found in the node list. Exiting."
    exit 1
fi

# ============================
#      DERIVE CONTROLLER HOST
# ============================

# The controller runs on the UE (Layer 0) node, so pull its IP straight from the
# node list instead of hardcoding it.
CONTROLLER_HOST="${layer_ip_map[0]}"
log "Controller host (Layer 0 / UE) resolved from node list: $CONTROLLER_HOST"
sleep 1

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
        sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no "$node_type@$ip" "kill -9 $pids_formatted"
        log "Killed $script with PID(s): $pids_formatted on Node '$node_type' ($ip)" | tee -a "$LOG_DIR/${node_log}"
    else
        log "No running instance of $script found on Node '$node_type' ($ip)." | tee -a "$LOG_DIR/${node_log}"
    fi
}


# Function to configure a single layer on a node
configure_layer() {
    local ip="$1"
    local layer="$2"
    local node_type="$3"

    local node_log="${node_type}_${ip}.log"
    log "--------------------------------------------" | tee -a "$LOG_DIR/${node_log}"
    log "Configuring Layer $layer on Node '$node_type' ($ip)" | tee -a "$LOG_DIR/${node_log}"
    log "--------------------------------------------" | tee -a "$LOG_DIR/${node_log}"
    sleep 1

    # Get the assigned port for this layer
    local assigned_port="${layer_port_map[$layer]}"

    # We always use the same script
    script="resilientNode.py"

    log "Selected script: $script" | tee -a "$LOG_DIR/${node_log}"
    sleep 1

    # Generate next layers indices
    next_layers=()
    for i in $(seq 1 5); do
        next_layer=$(((layer + i) % N))
        next_layers+=("$next_layer")
        if [ "$next_layer" -eq 0 ]; then
            break
        fi
    done

    # Build next_layers ip:port pairs
    next_layers_ips_ports=()
    for nl in "${next_layers[@]}"; do
        nl_ip="${layer_ip_map[$nl]}"
        nl_port="${layer_port_map[$nl]}"
        if [ -z "$nl_ip" ] || [ -z "$nl_port" ]; then
            log "ERROR: Next layer $nl details missing for Layer $layer on Node '$node_type' ($ip)." | tee -a "$LOG_DIR/${node_log}"
            return
        fi
        next_layers_ips_ports+=("$nl_ip:$nl_port")
    done

    log "Layer $layer will connect to next layers: ${next_layers_ips_ports[*]}." | tee -a "$LOG_DIR/${node_log}"
    sleep 1

    # SSH into the node and execute commands
    sshpass -p "$SSH_PASSWORD" ssh -o StrictHostKeyChecking=no "$node_type@$ip" 'bash -s' << EOF >> "$LOG_DIR/${node_log}" 2>&1
echo "Successfully logged into $node_type@$ip"

echo "Setting LED on"
echo 0 | sudo tee /sys/class/leds/ACT/brightness > /dev/null

echo "Navigating to rasp_compression directory..."
cd rasp_compression || { echo "ERROR: Directory not found! Exiting."; exit 1; }

echo "Activating virtual environment..."
source ./venv/bin/activate
if [ \$? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment. Exiting."
    exit 1
fi
echo "Virtual environment activated."

echo "Checking if all packages from requirements.txt are installed..."
missing_packages=\$(pip install -r requirements.txt --dry-run 2>&1 | grep "Collecting")
if [ -z "\$missing_packages" ]; then
    echo "All required packages are already installed."
else
    echo "The following packages are missing and need to be installed:"
    echo "\$missing_packages"
    echo "Installing missing required packages from requirements.txt..."
    pip install -r requirements.txt -qqq > /dev/null 2>&1
    if [ \$? -ne 0 ]; then
        echo "ERROR: Failed to install packages. Exiting."
        exit 1
    fi
    echo "Packages installed."
fi

echo "Running $script with layer $layer on port $assigned_port..."
nohup python "$script" --layer "$layer" --host "$ip" --port "$assigned_port" \
    --next_layers ${next_layers_ips_ports[@]} \
    --compression_method "$COMPRESSION_METHOD" \
    --compression_ratio "$COMPRESSION_RATIO" \
    --llmint8_outlier_ratio "$LLMINT8_OUTLIER_RATIO" \
    --llmint8_outlier_prec "$LLMINT8_OUTLIER_PREC" \
    --llmint8_regular_prec "$LLMINT8_REGULAR_PREC" \
    --controller_host "$CONTROLLER_HOST" \
    --controller_port "$CONTROLLER_PORT" \
    --control_base_port "$CONTROL_BASE_PORT" \
    </dev/null >/dev/null 2>&1 &
if [ \$? -ne 0 ]; then
    echo "ERROR: Failed to run $script. Exiting."
    exit 1
fi
echo "Script $script is running in the background at $node_type@$ip on port $assigned_port."

echo "Configuration for Layer $layer on Node '$node_type' ($ip) is complete. Closing SSH session."
EOF

    log "Completed configuration for Layer $layer on Node '$node_type' ($ip). Logs available at '$LOG_DIR/${node_log}'."
}

# Function to get local IP addresses
get_local_ips() {
    # Initialize an empty array
    local_ips=()

    if command -v ip >/dev/null 2>&1; then
        # Linux systems with ip command
        local_ips=$(ip addr show | awk '/inet / {print $2}' | cut -d'/' -f1)
    elif command -v ifconfig >/dev/null 2>&1; then
        # Systems with ifconfig command
        local_ips=$(ifconfig | awk '/inet / {print $2}' | grep -v '127.0.0.1')
    else
        # Fallback to hostname
        local_ip=$(hostname -i 2>/dev/null)
        if [ -n "$local_ip" ]; then
            local_ips="$local_ip"
        else
            log "ERROR: Unable to determine local IP addresses."
            exit 1
        fi
    fi
    echo "$local_ips"
}

# Function to configure the UE node (Layer 0)
configure_ue_node() {
    local ue_ip="$1"
    local ue_user="$2"

    local ue_log="${ue_user}_${ue_ip}.log"
    log "--------------------------------------------" | tee -a "$LOG_DIR/${ue_log}"
    log "Configuring UE Node '$ue_user' ($ue_ip)" | tee -a "$LOG_DIR/${ue_log}"
    log "--------------------------------------------" | tee -a "$LOG_DIR/${ue_log}"
    sleep 2  # Wait for 2 seconds

    # Specific script for Layer 0
    script="UX.py"

    log "Selected script: $script" | tee -a "$LOG_DIR/${ue_log}"
    sleep 2  # Wait for 2 seconds

    layer=0  # For the UE node, layer is 0

    # Generate next layers indices
    next_layers=()
    for i in {1..5}; do
        next_layer=$(( (layer + i) % N ))
        next_layers+=("$next_layer")
    done

    # Build next_layers ip:port pairs
    next_layers_ips_ports=()
    for nl in "${next_layers[@]}"; do
        nl_ip="${layer_ip_map[$nl]}"
        nl_port="${layer_port_map[$nl]}"
        if [ -z "$nl_ip" ] || [ -z "$nl_port" ]; then
            log "ERROR: Next layer $nl details missing for Layer $layer on Node '$ue_user' ($ue_ip)." | tee -a "$LOG_DIR/${ue_log}"
            return
        fi
        next_layers_ips_ports+=("$nl_ip:$nl_port")
    done

    log "Layer 0 will connect to next layers: ${next_layers_ips_ports[*]}." | tee -a "$LOG_DIR/${ue_log}"
    sleep 2  # Wait for 2 seconds

    # Get local IP addresses
    local current_ips=$(get_local_ips)

    # Check if the current machine is the UE node
    if echo "$current_ips" | tr ' ' '\n' | grep -w "$ue_ip" > /dev/null; then
        log "Running configuration locally on the UE node ($ue_ip)." | tee -a "$LOG_DIR/${ue_log}"

        # Kill the specific script if it's running locally
        pids=$(pgrep -f "$script")
        if [ -n "$pids" ]; then
            # Replace newlines with spaces and remove trailing space
            pids_formatted=$(echo "$pids" | tr '\n' ' ' | sed 's/[[:space:]]*$//')
            log "Found running $script with PID(s): $pids_formatted on local UE node ($ue_ip)" | tee -a "$LOG_DIR/${ue_log}"
            # Kill the process(es)
            kill -9 $pids_formatted
            log "Killed $script with PID(s): $pids_formatted on local UE node ($ue_ip)" | tee -a "$LOG_DIR/${ue_log}"
        else
            log "No running instance of $script found on local UE node ($ue_ip)." | tee -a "$LOG_DIR/${ue_log}"
        fi

        # Navigate to the rasp_compression directory
        log "Navigating to rasp_compression directory..." | tee -a "$LOG_DIR/${ue_log}"
        cd ~ || { log "ERROR: Unable to navigate to home directory! Exiting." | tee -a "$LOG_DIR/${ue_log}"; exit 1; }
        cd RESEARCH/rasp_compression || { log "ERROR: Directory rasp_compression not found in home directory! Exiting." | tee -a "$LOG_DIR/${ue_log}"; exit 1; }


        # Activate virtual environment
        log "Activating virtual environment..." | tee -a "$LOG_DIR/${ue_log}"
        source ./venv/bin/activate
        if [ $? -ne 0 ]; then
            log "ERROR: Failed to activate virtual environment. Exiting." | tee -a "$LOG_DIR/${ue_log}"
            exit 1
        fi
        log "Virtual environment activated." | tee -a "$LOG_DIR/${ue_log}"

        # Check for missing packages
        log "Checking if all packages from requirements.txt are installed..." | tee -a "$LOG_DIR/${ue_log}"
        missing_packages=$(pip install -r requirements.txt --dry-run 2>&1 | grep "Collecting")
        if [ -z "$missing_packages" ]; then
            log "All required packages are already installed." | tee -a "$LOG_DIR/${ue_log}"
        else
            log "The following packages are missing and need to be installed:" | tee -a "$LOG_DIR/${ue_log}"
            log "$missing_packages" | tee -a "$LOG_DIR/${ue_log}"
            log "Installing missing required packages from requirements.txt..." | tee -a "$LOG_DIR/${ue_log}"
            pip install -r requirements.txt -qqq > /dev/null 2>&1
            if [ $? -ne 0 ]; then
                log "ERROR: Failed to install packages. Exiting." | tee -a "$LOG_DIR/${ue_log}"
                exit 1
            fi
            log "Packages installed." | tee -a "$LOG_DIR/${ue_log}"
        fi

        # Run the script
        log "Running $script with layer 0..." | tee -a "$LOG_DIR/${ue_log}"
        streamlit run "$script" -- --host "$ue_ip" --port "${layer_port_map[0]}" --next_layers "${next_layers_ips_ports[@]}" #> /dev/null 2>&1 &
        if [ $? -ne 0 ]; then
            log "ERROR: Failed to run $script. Exiting." | tee -a "$LOG_DIR/${ue_log}"
            exit 1
        fi
        log "Script $script is running." | tee -a "$LOG_DIR/${ue_log}"

        log "Completed configuration for UE Node '$ue_user' ($ue_ip). Logs available at '$LOG_DIR/${ue_log}'." | tee -a "$LOG_DIR/${ue_log}"
    else
        log "SSH into UE node ($ue_ip) to run configuration." | tee -a "$LOG_DIR/${ue_log}"

         # Kill the specific script if it's running
        kill_script "$ue_ip" "$ue_user" "$script" "$ue_log"

        # SSH into the UE node with tty allocation and execute commands
        sshpass -p "$SSH_PASSWORD" ssh -t -o StrictHostKeyChecking=no "$ue_user@$ue_ip" bash -s << EOF | tee -a "$LOG_DIR/${ue_log}"
echo "Successfully logged into $ue_user@$ue_ip"

echo "Navigating to rasp_compression directory..."
cd rasp_compression || { echo "ERROR: Directory not found! Exiting."; exit 1; }

echo "Activating virtual environment..."
source ./venv/bin/activate
if [ \$? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment. Exiting."
    exit 1
fi
echo "Virtual environment activated."

echo "Checking if all packages from requirements.txt are installed..."
missing_packages=\$(pip install -r requirements.txt --dry-run 2>&1 | grep "Collecting")
if [ -z "\$missing_packages" ]; then
    echo "All required packages are already installed."
else
    echo "The following packages are missing and need to be installed:"
    echo "\$missing_packages"
    echo "Installing missing required packages from requirements.txt..."
    pip install -r requirements.txt -qqq > /dev/null 2>&1
    if [ \$? -ne 0 ]; then
        echo "ERROR: Failed to install packages. Exiting."
        exit 1
    fi
    echo "Packages installed."
fi

echo "Running $script with layer 0..."
streamlit run "$script" -- --host "$ue_ip" --port "${layer_port_map[0]}" --next_layers "${next_layers_ips_ports[@]}"
if [ \$? -ne 0 ]; then
    echo "ERROR: Failed to run $script. Exiting."
    exit 1
fi
echo "Script $script is running."

EOF

        log "Completed configuration for UE Node '$ue_user' ($ue_ip). Logs available at '$LOG_DIR/${ue_log}'." | tee -a "$LOG_DIR/${ue_log}"
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
    sleep 2  # Wait for 2 seconds
done

# ============================
#    CONFIGURE REGULAR NODES
# ============================

for entry in "${combined_entries[@]}"; do
    ip=$(echo "$entry" | cut -d'-' -f1)
    layer=$(echo "$entry" | cut -d'-' -f2)
    node_type=${node_user_map["$ip"]}

    # Determine if this is the UE node (layer 0)
    if [ "$layer" -eq 0 ]; then
        configure_ue_node "$ip" "$node_type" 
    else
        configure_layer "$ip" "$layer" "$node_type" &
    fi

    # Sleep interval between configurations to avoid overwhelming the system
    sleep 6
done


log "Script execution finalized."
