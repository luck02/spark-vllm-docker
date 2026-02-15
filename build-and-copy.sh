#!/bin/bash
set -e

# Start total time tracking
START_TIME=$(date +%s)

# Default values
IMAGE_TAG="vllm-node"
REBUILD_DEPS=false
REBUILD_VLLM=false
COPY_HOSTS=()
SSH_USER="$USER"
NO_BUILD=false
TRITON_REF="v3.6.0"
VLLM_REF="main"
TMP_IMAGE=""
PARALLEL_COPY=false
USE_WHEELS_MODE=""
PRE_FLASHINFER=false
PRE_TRANSFORMERS=false
EXP_MXFP4=false
TRITON_REF_SET=false
VLLM_REF_SET=false
VLLM_PRS=""
FULL_LOG=false

cleanup() {
    if [ -n "$TMP_IMAGE" ] && [ -f "$TMP_IMAGE" ]; then
        echo "Cleaning up temporary image $TMP_IMAGE"
        rm -f "$TMP_IMAGE"
    fi
}

trap cleanup EXIT

add_copy_hosts() {
    local token part
    for token in "$@"; do
        IFS=',' read -ra PARTS <<< "$token"
        for part in "${PARTS[@]}"; do
            part="${part//[[:space:]]/}"
            if [ -n "$part" ]; then
                COPY_HOSTS+=("$part")
            fi
        done
    done
}

copy_to_host() {
    local host="$1"
    echo "Loading image into ${SSH_USER}@${host}..."
    local host_copy_start host_copy_end host_copy_time
    host_copy_start=$(date +%s)
    if cat "$TMP_IMAGE" | ssh "${SSH_USER}@${host}" "docker load"; then
        host_copy_end=$(date +%s)
        host_copy_time=$((host_copy_end - host_copy_start))
        printf "Copy to %s completed in %02d:%02d:%02d\n" "$host" $((host_copy_time/3600)) $((host_copy_time%3600/60)) $((host_copy_time%60))
    else
        echo "Copy to $host failed."
        return 1
    fi
}
BUILD_JOBS="16"
GPU_ARCH_LIST="12.1a"

# Help function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -t, --tag <tag>           : Image tag (default: 'vllm-node')"
    echo "  --gpu-arch <arch>         : GPU architecture (default: '12.1a')"
    echo "  --rebuild-deps            : Set cache bust for dependencies"
    echo "  --rebuild-vllm            : Set cache bust for vllm"
    echo "  --triton-ref <ref>        : Triton commit SHA, branch or tag (default: 'v3.5.1')"
    echo "  --vllm-ref <ref>          : vLLM commit SHA, branch or tag (default: 'main')"
    echo "  -c, --copy-to <hosts>     : Host(s) to copy the image to. Accepts comma or space-delimited lists after the flag."
    echo "      --copy-to-host        : Alias for --copy-to (backwards compatibility)."
    echo "      --copy-parallel       : Copy to all hosts in parallel instead of serially."
    echo "  -j, --build-jobs <jobs>   : Number of concurrent build jobs (default: \${BUILD_JOBS})"
    echo "  -u, --user <user>         : Username for ssh command (default: \$USER)"
    echo "  --use-wheels [mode]       : Use prebuilt vLLM wheels. Mode can be 'nightly' (default) or 'release'."
    echo "  --pre-flashinfer          : Use pre-release versions of FlashInfer"
    echo "  --pre-tf, --pre-transformers : Install transformers 5.0.0rc0 or higher"
    echo "  --exp-mxfp4, --experimental-mxfp4 : Build with experimental native MXFP4 support"
    echo "  --apply-vllm-pr <pr-num>  : Apply a specific PR patch to vLLM source code. Can be specified multiple times."
    echo "  --full-log                : Enable full build logging (--progress=plain)"
    echo "  --no-build                : Skip building, only copy image (requires --copy-to)"
    echo "  -h, --help                : Show this help message"
    exit 1
}

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--tag) IMAGE_TAG="$2"; shift ;;
        --gpu-arch) GPU_ARCH_LIST="$2"; shift ;;
        --rebuild-deps) REBUILD_DEPS=true ;;
        --rebuild-vllm) REBUILD_VLLM=true ;;
        --triton-ref) TRITON_REF="$2"; TRITON_REF_SET=true; shift ;;
        --vllm-ref) VLLM_REF="$2"; VLLM_REF_SET=true; shift ;;
        -c|--copy-to|--copy-to-host|--copy-to-hosts)
            shift
            # Consume arguments until the next flag or end of args
            while [[ "$#" -gt 0 && "$1" != -* ]]; do
                add_copy_hosts "$1"
                shift
            done

            # If no hosts specified, use autodiscovery
            if [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
                echo "No hosts specified. Using autodiscovery..."
                source "$(dirname "$0")/autodiscover.sh"
                
                detect_nodes
                if [ $? -ne 0 ]; then
                    echo "Error: Autodiscovery failed."
                    exit 1
                fi
                
                # Use PEER_NODES directly
                if [ ${#PEER_NODES[@]} -gt 0 ]; then
                    COPY_HOSTS=("${PEER_NODES[@]}")
                fi
                
                if [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
                     echo "Error: Autodiscovery found no other nodes."
                     exit 1
                fi
                echo "Autodiscovered hosts: ${COPY_HOSTS[*]}"
            fi
            continue
            ;;
        -j|--build-jobs) BUILD_JOBS="$2"; shift ;;
        -u|--user) SSH_USER="$2"; shift ;;
        --copy-parallel) PARALLEL_COPY=true ;;
        --use-wheels)
            if [[ "$2" != -* && -n "$2" ]]; then
                if [[ "$2" != "nightly" && "$2" != "release" ]]; then
                    echo "Error: --use-wheels argument must be 'nightly' or 'release'."
                    exit 1
                fi
                USE_WHEELS_MODE="$2"
                shift
            else
                USE_WHEELS_MODE="nightly"
            fi
            ;;
        --pre-flashinfer) PRE_FLASHINFER=true ;;
        --pre-tf|--pre-transformers) PRE_TRANSFORMERS=true ;;
        --exp-mxfp4|--experimental-mxfp4) EXP_MXFP4=true ;;
        --apply-vllm-pr)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
               if [ -n "$VLLM_PRS" ]; then
                   VLLM_PRS="$VLLM_PRS $2"
               else
                   VLLM_PRS="$2"
               fi
               shift
            else
               echo "Error: --apply-vllm-pr requires a PR number."
               exit 1
            fi
            ;;
        --full-log) FULL_LOG=true ;;
        --no-build) NO_BUILD=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -n "$VLLM_PRS" ]; then
    if [ "$EXP_MXFP4" = true ]; then echo "Error: --apply-vllm-pr is incompatible with --exp-mxfp4"; exit 1; fi
    if [ -n "$USE_WHEELS_MODE" ]; then echo "Error: --apply-vllm-pr is incompatible with --use-wheels"; exit 1; fi
fi

if [ "$EXP_MXFP4" = true ]; then
    if [ "$TRITON_REF_SET" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --triton-ref"; exit 1; fi
    if [ "$VLLM_REF_SET" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --vllm-ref"; exit 1; fi
    if [ -n "$USE_WHEELS_MODE" ]; then echo "Error: --exp-mxfp4 is incompatible with --use-wheels"; exit 1; fi
    if [ "$PRE_FLASHINFER" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --pre-flashinfer"; exit 1; fi
    if [ "$PRE_TRANSFORMERS" = true ]; then echo "Error: --exp-mxfp4 is incompatible with --pre-transformers"; exit 1; fi
fi

if [ -n "$USE_WHEELS_MODE" ]; then
    read -p "!!! Wheels build is known not to work properly with all models after migration to Torch 2.10! Full build is recommended. Do you want to continue (y/N)? " choice
    case "$choice" in 
        y|Y ) echo "Continuing...";;
        * ) echo "Aborting."; exit 1;;
    esac
fi

# Validate --no-build usage
if [ "$NO_BUILD" = true ] && [ "${#COPY_HOSTS[@]}" -eq 0 ]; then
    echo "Error: --no-build requires --copy-to to be specified"
    exit 1
fi

# Build image (unless --no-build is set)
BUILD_TIME=0
if [ "$NO_BUILD" = false ]; then
    # Construct build command
    CMD=("docker" "build" "-t" "$IMAGE_TAG")

    if [ "$FULL_LOG" = true ]; then
        CMD+=("--progress=plain")
    fi

    if [ "$EXP_MXFP4" = true ]; then
        echo "Building with experimental MXFP4 support..."
        CMD+=("-f" "Dockerfile.mxfp4")
    elif [ -n "$USE_WHEELS_MODE" ]; then
        echo "Using pre-built vLLM wheels (mode: $USE_WHEELS_MODE)"
        CMD+=("-f" "Dockerfile.wheels")
        if [ "$USE_WHEELS_MODE" = "release" ]; then
             CMD+=("--build-arg" "WHEELS_FROM_GITHUB_RELEASE=1")
        fi
    else
        echo "Building vLLM from source"
    fi

    if [ "$REBUILD_DEPS" = true ]; then
        echo "Setting CACHEBUST_DEPS..."
        CMD+=("--build-arg" "CACHEBUST_DEPS=$(date +%s)")
    fi

    if [ "$REBUILD_VLLM" = true ]; then
        echo "Setting CACHEBUST_VLLM..."
        CMD+=("--build-arg" "CACHEBUST_VLLM=$(date +%s)")
    fi

    # Add TRITON_REF to build arguments
    CMD+=("--build-arg" "TRITON_REF=$TRITON_REF")

    # Add VLLM_REF to build arguments
    CMD+=("--build-arg" "VLLM_REF=$VLLM_REF")

    # Add BUILD_JOBS to build arguments
    CMD+=("--build-arg" "BUILD_JOBS=$BUILD_JOBS")

    # Add GPU architecture to build arguments
    CMD+=("--build-arg" "TORCH_CUDA_ARCH_LIST=$GPU_ARCH_LIST")
    CMD+=("--build-arg" "FLASHINFER_CUDA_ARCH_LIST=$GPU_ARCH_LIST")

    if [ "$PRE_FLASHINFER" = true ]; then
        echo "Using pre-release FlashInfer..."
        CMD+=("--build-arg" "FLASHINFER_PRE=--pre")
    fi

    if [ -n "$VLLM_PRS" ]; then
        echo "Applying vLLM PRs: $VLLM_PRS"
        CMD+=("--build-arg" "VLLM_PRS=$VLLM_PRS")
    fi

    if [ "$PRE_TRANSFORMERS" = true ]; then
        echo "Using transformers>=5.0.0..."
        CMD+=("--build-arg" "PRE_TRANSFORMERS=1")
    fi

    # Add build context
    CMD+=(".")

    # Execute build
    echo "Building image with command: ${CMD[*]}"
    BUILD_START=$(date +%s)
    "${CMD[@]}"
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
else
    echo "Skipping build (--no-build specified)"
fi

# Copy to host if requested
COPY_TIME=0
if [ "${#COPY_HOSTS[@]}" -gt 0 ]; then
    echo "Copying image '$IMAGE_TAG' to ${#COPY_HOSTS[@]} host(s): ${COPY_HOSTS[*]}"
    if [ "$PARALLEL_COPY" = true ]; then
        echo "Parallel copy enabled."
    fi
    COPY_START=$(date +%s)

    TMP_IMAGE=$(mktemp -t vllm_image.XXXXXX)
    echo "Saving image locally to $TMP_IMAGE..."
    docker save -o "$TMP_IMAGE" "$IMAGE_TAG"

    if [ "$PARALLEL_COPY" = true ]; then
        PIDS=()
        for host in "${COPY_HOSTS[@]}"; do
            copy_to_host "$host" &
            PIDS+=($!)
        done
        COPY_FAILURE=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                COPY_FAILURE=1
            fi
        done
        if [ "$COPY_FAILURE" -ne 0 ]; then
            echo "One or more copies failed."
            exit 1
        fi
    else
        for host in "${COPY_HOSTS[@]}"; do
            copy_to_host "$host"
        done
    fi

    COPY_END=$(date +%s)
    COPY_TIME=$((COPY_END - COPY_START))
    echo "Copy complete."
else
    echo "No host specified, skipping copy."
fi

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# Display timing statistics
echo ""
echo "========================================="
echo "         TIMING STATISTICS"
echo "========================================="
if [ "$BUILD_TIME" -gt 0 ]; then
    echo "Docker Build:  $(printf '%02d:%02d:%02d' $((BUILD_TIME/3600)) $((BUILD_TIME%3600/60)) $((BUILD_TIME%60)))"
fi
if [ "$COPY_TIME" -gt 0 ]; then
    echo "Image Copy:    $(printf '%02d:%02d:%02d' $((COPY_TIME/3600)) $((COPY_TIME%3600/60)) $((COPY_TIME%60)))"
fi
echo "Total Time:    $(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))"
echo "========================================="
echo "Done building $IMAGE_TAG."
