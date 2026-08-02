#!/bin/bash
# Chain: T1.4 retry (current image) → cluster stop → rebuild vllm-node → T2.1 (new image)
# Started manually under nohup. Each phase logs separately.

set +e  # keep going on partial failure
TS=$(date +%Y%m%d-%H%M%S)
CHAIN_LOG=~/dev/spark-vllm-docker/tuning/retry-chain-$TS.log
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "===== CHAIN START $(date -u +%FT%TZ) ====="

# --- Phase 1: T1.4 on current image ---
echo "===== [1/4] T1.4 sweep start $(date -u +%FT%TZ) ====="
cd ~/dev/spark-vllm-docker/tuning
python3 sweep.py --only T1.4
T14_RC=$?
echo "===== [1/4] T1.4 sweep done rc=$T14_RC $(date -u +%FT%TZ) ====="

# --- Phase 2: stop both nodes cleanly to free memory for build ---
echo "===== [2/4] Cluster stop $(date -u +%FT%TZ) ====="
cd ~/dev/spark-vllm-docker
./launch-cluster.sh stop || true
docker rm -f vllm_node 2>/dev/null || true
echo "===== [2/4] Cluster stopped $(date -u +%FT%TZ) ====="

# --- Phase 3: rebuild vllm-node image + copy to spark2 ---
echo "===== [3/4] Image rebuild start $(date -u +%FT%TZ) ====="
cd ~/dev/spark-vllm-docker
git pull --ff-only origin main || echo "git pull warning (non-fatal)"
./build-and-copy.sh -t vllm-node -c spark2 --copy-parallel
BUILD_RC=$?
echo "===== [3/4] Image rebuild done rc=$BUILD_RC $(date -u +%FT%TZ) ====="
docker inspect vllm-node --format "{{.Created}}" || echo "no vllm-node image!"

# --- Phase 4: T2.1 on new image ---
echo "===== [4/4] T2.1 sweep start $(date -u +%FT%TZ) ====="
cd ~/dev/spark-vllm-docker/tuning
python3 sweep.py --only T2.1
T21_RC=$?
echo "===== [4/4] T2.1 sweep done rc=$T21_RC $(date -u +%FT%TZ) ====="

echo "===== CHAIN COMPLETE $(date -u +%FT%TZ) ====="
echo "T1.4=$T14_RC BUILD=$BUILD_RC T2.1=$T21_RC"
