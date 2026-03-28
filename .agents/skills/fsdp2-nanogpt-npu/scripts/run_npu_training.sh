#!/bin/bash
# run_npu_training.sh — Launch FSDP2 nanoGPT training on Ascend NPU
#
# Usage:
#   bash run_npu_training.sh [num_npus] [extra_args...]
#
# Examples:
#   bash run_npu_training.sh 2                    # 2-card training
#   bash run_npu_training.sh 4 --mixed-precision  # 4-card with bf16
#   bash run_npu_training.sh 2 --dcp-api          # DCP checkpoint API

set -euo pipefail

NUM_NPUS=${1:-2}
shift 2>/dev/null || true
EXTRA_ARGS="$@"

# Activate conda environment
CONDA_ENV=${FSDP2_CONDA_ENV:-fsdp2_npu}
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
    echo "Activated conda env: ${CONDA_ENV}"
fi

# CANN environment
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# Build device list: 0,1,...,N-1
DEVICES=$(seq -s, 0 $((NUM_NPUS - 1)))
export ASCEND_RT_VISIBLE_DEVICES=${DEVICES}

echo "============================================"
echo " FSDP2 nanoGPT NPU Training"
echo " NPUs: ${NUM_NPUS} (devices: ${DEVICES})"
echo " Extra args: ${EXTRA_ARGS:-none}"
echo "============================================"

# Verify NPU availability
python3 -c "
import torch
import torch_npu
count = torch.npu.device_count()
assert count >= ${NUM_NPUS}, f'Need ${NUM_NPUS} NPUs but only {count} available'
print(f'NPU check passed: {count} devices available')
"

echo ""
echo ">>> Starting training run..."
torchrun --nnodes=1 --nproc_per_node=${NUM_NPUS} example.py ${EXTRA_ARGS}

echo ""
echo ">>> Training complete. Checking checkpoints..."
if [ -d "checkpoints" ]; then
    CKPT_COUNT=$(find checkpoints/ -type d -mindepth 2 -maxdepth 2 | wc -l)
    echo "Found ${CKPT_COUNT} checkpoint(s) in checkpoints/"
    ls -la checkpoints/dtensor_api/ 2>/dev/null || ls -la checkpoints/ 2>/dev/null
else
    echo "WARNING: No checkpoints directory found"
fi
