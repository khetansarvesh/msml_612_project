#!/bin/bash
# Run single-GPU training
# Use this script for g4dn.2xlarge or g4dn.xlarge instances

set -e

echo "🚀 Starting Single-GPU Training..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check GPU
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -lt 1 ]; then
    echo "❌ No GPUs found!"
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  GPUs: $GPU_COUNT"
echo "  Mode: Single-GPU (no DDP)"
echo "  Working Directory: $(pwd)"
echo ""

# Check for existing checkpoints
if [ -f "model/latest_checkpoint.pth" ]; then
    echo -e "${GREEN}✓ Found checkpoint - will auto-resume${NC}"
elif [ -d "model" ] && [ "$(ls -A model/*.pth 2>/dev/null)" ]; then
    echo -e "${YELLOW}⚠ Found old checkpoints${NC}"
    ls -lh model/*.pth
else
    echo -e "${YELLOW}Starting fresh training${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Training...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Run with torchrun but only 1 GPU
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=29500 \
    run.py 2>&1 | tee -a training.log

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Outputs:"
echo "  - Checkpoints: model/"
echo "  - Training log: training.log"
echo "  - Loss plot: outputs/training_loss.png"
echo ""
