#!/bin/bash
# Run DDP Training with 4 GPUs using torchrun
# This script automatically resumes from latest checkpoint if available

set -e

echo "🚀 Starting DDP Training with 4 GPUs..."

# Configuration
NUM_GPUS=4
MASTER_PORT=29500

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if GPUs are available
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo "⚠ Warning: Found only $GPU_COUNT GPUs, but configured for $NUM_GPUS"
    echo "Adjusting to use $GPU_COUNT GPUs..."
    NUM_GPUS=$GPU_COUNT
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  GPUs: $NUM_GPUS"
echo "  Master Port: $MASTER_PORT"
echo "  Working Directory: $(pwd)"
echo ""

# Check for existing checkpoints
if [ -f "model/latest_checkpoint.pth" ]; then
    echo -e "${GREEN}✓ Found latest checkpoint - will auto-resume${NC}"
elif [ -d "model" ] && [ "$(ls -A model/*.pth 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠ Found checkpoints but no latest_checkpoint.pth${NC}"
    echo "  Available checkpoints:"
    ls -lh model/*.pth
else
    echo -e "${YELLOW}Starting fresh training (no checkpoints found)${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Training...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Run training with torchrun
# torchrun automatically sets up the distributed environment
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    run.py 2>&1 | tee -a training.log

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Outputs saved to:"
echo "  - Checkpoints: model/"
echo "  - Samples: outputs/"
echo "  - Training log: training.log"
echo ""
