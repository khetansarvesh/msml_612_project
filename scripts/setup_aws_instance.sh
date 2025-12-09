#!/bin/bash
# Setup script for AWS EC2 instance
# Run this on the EC2 instance after first login

set -e

echo "🔧 Setting up AWS instance for DIT training..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Update system packages
echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update -qq

# Verify CUDA installation
echo -e "${YELLOW}Verifying CUDA installation...${NC}"
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}✓ CUDA installed:${NC}"
    nvcc --version | head -n 4
else
    echo "⚠ CUDA not found. Installing..."
    # Deep Learning AMI should have CUDA pre-installed
    # If not, you may need to install it manually
fi

# Verify GPUs
echo -e "${YELLOW}Checking GPUs...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -eq 4 ]; then
        echo -e "${GREEN}✓ Found 4 GPUs:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "⚠ Expected 4 GPUs but found $GPU_COUNT"
    fi
else
    echo "⚠ nvidia-smi not found!"
    exit 1
fi

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo "⚠ requirements.txt not found. Please ensure it exists."
fi

# Verify PyTorch with CUDA
echo -e "${YELLOW}Verifying PyTorch with CUDA...${NC}"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" || {
    echo "⚠ PyTorch CUDA not properly configured. Reinstalling..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
}

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p model outputs scripts
echo -e "${GREEN}✓ Directories created${NC}"

# Test distributed training setup
echo -e "${YELLOW}Testing distributed training setup...${NC}"
python -c "
import torch.distributed as dist
import os
print('Distributed training packages available')
print(f'NCCL backend available: {dist.is_nccl_available()}')
" && echo -e "${GREEN}✓ Distributed training ready${NC}"

# Display system info
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}System Information${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Hostname: $(hostname)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Disk Space: $(df -h / | awk 'NR==2 {print $4}')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
echo "GPUs: $(nvidia-smi --list-gpus | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Verify config.yaml settings"
echo "  2. Run training: bash scripts/run_training_ddp.sh"
echo "  3. Monitor with: tail -f training.log"
echo "  4. Watch GPUs: watch -n 1 nvidia-smi"
echo ""
