#!/bin/bash

# Default Configuration
IMAGE_NAME="vllm-node"
DEFAULT_CONTAINER_NAME="vllm_node"
ETH_IF="enp1s0f1np1"
IB_IF="rocep1s0f1,roceP2p1s0f1"

# Initialize variables
NODES_ARG=""
CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
COMMAND_TO_RUN=""
DAEMON_MODE="false"
ACTION="start"

# Function to print usage
usage() {
    echo "Usage: $0 -n <node_ips> [-t <image_name>] [--name <container_name>] [--eth-if <if_name>] [--ib-if <if_name>] [-d] [action] [command]"
    echo "  -n, --nodes     Comma-separated list of node IPs (Mandatory)"
    echo "  -t              Docker image name (Optional, default: $IMAGE_NAME)"
    echo "  --name          Container name (Optional, default: $DEFAULT_CONTAINER_NAME)"
    echo "  --eth-if        Ethernet interface (Optional, default: $ETH_IF)"
    echo "  --ib-if         InfiniBand interface (Optional, default: $IB_IF)"
    echo "  -d              Daemon mode (only for 'start' action)"
    echo "  action          start | stop | status | exec (Default: start)"
    echo "  command         Command to run (only for 'exec' action)"
    exit 1
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--nodes) NODES_ARG="$2"; shift ;;
        -t) IMAGE_NAME="$2"; shift ;;
        --name) CONTAINER_NAME="$2"; shift ;;
        --eth-if) ETH_IF="$2"; shift ;;
        --ib-if) IB_IF="$2"; shift ;;
        -d) DAEMON_MODE="true" ;;
        -h|--help) usage ;;
        start|stop|status) 
            ACTION="$1" 
            ;;
        exec)
            ACTION="exec"
            shift
            COMMAND_TO_RUN="$@"
            break
            ;;
        *) 
            # If it's not a flag and not a known action, treat as exec command for backward compatibility
            # unless it's the default 'start' implied.
            # However, to support "omitted" = start, we need to be careful.
            # If the arg looks like a command, it's exec.
            ACTION="exec"
            COMMAND_TO_RUN="$@"
            break 
            ;;
    esac
    shift
done

if [[ -z "$NODES_ARG" ]]; then
    echo "Error: Nodes argument (-n) is mandatory."
    usage
fi

# Split nodes into array
IFS=',' read -r -a ALL_NODES <<< "$NODES_ARG"

# Detect Head IP (Local IP)
HEAD_IP=""
LOCAL_IPS=$(hostname -I)
for ip in "${ALL_NODES[@]}"; do
    # Trim whitespace
    ip=$(echo "$ip" | xargs)
    if [[ " $LOCAL_IPS " =~ " $ip " ]]; then
        HEAD_IP="$ip"
        break
    fi
done

if [[ -z "$HEAD_IP" ]]; then
    echo "Error: Could not determine Head IP. This script must be run on one of the nodes specified in -n."
    exit 1
fi

# Identify Worker Nodes
WORKER_NODES=()
for ip in "${ALL_NODES[@]}"; do
    ip=$(echo "$ip" | xargs)
    if [[ "$ip" != "$HEAD_IP" ]]; then
        WORKER_NODES+=("$ip")
    fi
done

echo "Head Node: $HEAD_IP"
echo "Worker Nodes: ${WORKER_NODES[*]}"
echo "Container Name: $CONTAINER_NAME"
echo "Action: $ACTION"

# Cleanup Function
cleanup() {
    echo ""
    echo "Stopping cluster..."
    
    # Stop Head
    echo "Stopping head node ($HEAD_IP)..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
    
    # Stop Workers
    for worker in "${WORKER_NODES[@]}"; do
        echo "Stopping worker node ($worker)..."
        ssh "$worker" "docker stop $CONTAINER_NAME" >/dev/null 2>&1 || true
    done
    
    echo "Cluster stopped."
}

# Handle 'stop' action
if [[ "$ACTION" == "stop" ]]; then
    cleanup
    exit 0
fi

# Handle 'status' action
if [[ "$ACTION" == "status" ]]; then
    echo "Checking status..."
    
    # Check Head
    if docker ps | grep -q "$CONTAINER_NAME"; then
        echo "[HEAD] $HEAD_IP: Container '$CONTAINER_NAME' is RUNNING."
        echo "--- Ray Status ---"
        docker exec "$CONTAINER_NAME" ray status || echo "Failed to get ray status."
        echo "------------------"
    else
        echo "[HEAD] $HEAD_IP: Container '$CONTAINER_NAME' is NOT running."
    fi
    
    # Check Workers
    for worker in "${WORKER_NODES[@]}"; do
        if ssh "$worker" "docker ps | grep -q '$CONTAINER_NAME'"; then
             echo "[WORKER] $worker: Container '$CONTAINER_NAME' is RUNNING."
        else
             echo "[WORKER] $worker: Container '$CONTAINER_NAME' is NOT running."
        fi
    done
    exit 0
fi

# Trap signals
# Only trap if we are NOT in daemon mode, OR if we are in exec mode (always cleanup after exec)
if [[ "$DAEMON_MODE" == "false" ]] || [[ "$ACTION" == "exec" ]]; then
    trap cleanup EXIT INT TERM HUP
fi

# Start Head Node
echo "Starting Head Node on $HEAD_IP..."
docker run -d --privileged --gpus all --rm \
    --ipc=host --network host \
    --name "$CONTAINER_NAME" \
    -e NCCL_DEBUG=INFO -e NCCL_IGNORE_CPU_AFFINITY=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    "$IMAGE_NAME" \
    ./run-cluster-node.sh \
    --role head \
    --host-ip "$HEAD_IP" \
    --eth-if "$ETH_IF" \
    --ib-if "$IB_IF"

# Start Worker Nodes
for worker in "${WORKER_NODES[@]}"; do
    echo "Starting Worker Node on $worker..."
    ssh "$worker" "docker run -d --privileged --gpus all --rm \
        --ipc=host --network host \
        --name $CONTAINER_NAME \
        -e NCCL_DEBUG=INFO -e NCCL_IGNORE_CPU_AFFINITY=1 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        $IMAGE_NAME \
        ./run-cluster-node.sh \
        --role node \
        --host-ip $worker \
        --eth-if $ETH_IF \
        --ib-if $IB_IF \
        --head-ip $HEAD_IP"
done

# Wait for Cluster Readiness
wait_for_cluster() {
    echo "Waiting for cluster to be ready..."
    local retries=30
    local count=0
    
    while [[ $count -lt $retries ]]; do
        # Check if ray is responsive
        if docker exec "$CONTAINER_NAME" ray status >/dev/null 2>&1; then
             echo "Cluster head is responsive."
             # Give workers a moment to connect
             sleep 5
             return 0
        fi
        
        sleep 2
        ((count++))
    done
    
    echo "Timeout waiting for cluster to start."
    exit 1
}

if [[ "$ACTION" == "exec" ]]; then
    wait_for_cluster
    echo "Executing command: $COMMAND_TO_RUN"
    eval "$COMMAND_TO_RUN"
elif [[ "$ACTION" == "start" ]]; then
    wait_for_cluster
    if [[ "$DAEMON_MODE" == "true" ]]; then
        echo "Cluster started in background (Daemon mode)."
    else
        echo "Cluster started. Tailing logs from head node..."
        echo "Press Ctrl+C to stop the cluster."
        docker logs -f "$CONTAINER_NAME" &
        wait $!
    fi
fi
