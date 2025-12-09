# AWS Setup Guide for 4-GPU DDP Training

This guide walks you through setting up a **g4dn.12xlarge Spot instance** on AWS for training your Diffusion Transformer model with 4 GPUs.

## Prerequisites

- AWS account with $100 credits activated
- AWS CLI installed (`brew install awscli` or download from [aws.amazon.com/cli](https://aws.amazon.com/cli/))
- AWS credentials configured (`aws configure`)
- SSH key pair created in AWS Console

## Cost Summary

**Instance: g4dn.12xlarge (4x NVIDIA T4 GPUs)**

- **Spot price:** ~$1.17/hour
- **On-Demand price:** $3.912/hour (fallback if Spot unavailable)
- **Budget usage:** ~85 hours with $100 on Spot
- **Training time:** ~3.5-6.5 hours for 200 epochs

## Step-by-Step Setup

### 1. Create SSH Key Pair (One-time)

If you don't have an SSH key pair in AWS:

```bash
# Create key pair in AWS
aws ec2 create-key-pair \
  --key-name dit-training-key \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/dit-training-key.pem

# Set proper permissions
chmod 400 ~/.ssh/dit-training-key.pem
```

### 2. Launch Spot Instance

**Option A: Using the provided script (Recommended)**

```bash
cd /Users/sarveshkhetan/work/msml_612_project
bash scripts/launch_spot_instance.sh
```

This will output the instance ID and public IP address. Save these!

**Option B: Manual launch via AWS Console**

1. Go to EC2 Dashboard → Spot Requests
2. Click "Request Spot Instances"
3. Select AMI: **Deep Learning AMI GPU PyTorch 2.1.2 (Ubuntu 20.04)**
4. Instance type: **g4dn.12xlarge**
5. Max price: **$1.50/hour** (to ensure you get capacity)
6. Request type: **Persistent** (auto-restarts after interruption)
7. Security group: Allow SSH (port 22) from your IP
8. Key pair: Select your key pair
9. Storage: 100GB EBS (enough for model + checkpoints)

### 3. Connect to Instance

```bash
# Wait ~2-3 minutes for instance to start
# Replace <instance-ip> with your instance's public IP

ssh -i ~/.ssh/dit-training-key.pem ubuntu@<instance-ip>
```

### 4. Setup Instance

Once connected, run the setup script:

```bash
# Clone your repo (or upload via scp)
git clone <your-repo-url>
cd msml_612_project

# Run setup script
bash scripts/setup_aws_instance.sh
```

This will:

- Install Python dependencies
- Verify CUDA and PyTorch
- Check all 4 GPUs are available
- Create necessary directories

### 5. Verify GPU Setup

```bash
nvidia-smi
```

You should see **4 Tesla T4 GPUs** listed.

### 6. Start Training

```bash
# Start 4-GPU DDP training
bash scripts/run_training_ddp.sh
```

The script will:

- Automatically check for checkpoints and resume if found
- Run training with `torchrun` for DDP
- Log output to `training.log`
- Save checkpoints every 10 epochs

### 7. Monitor Training

**In another terminal (from your local machine):**

```bash
# SSH with port forwarding for TensorBoard (optional)
ssh -i ~/.ssh/dit-training-key.pem -L 6006:localhost:6006 ubuntu@<instance-ip>

# Watch training progress
tail -f ~/msml_612_project/training.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### 8. Handle Spot Interruptions

If your Spot instance is interrupted:

1. AWS will automatically restart it (persistent request)
2. SSH back in: `ssh -i ~/.ssh/dit-training-key.pem ubuntu@<instance-ip>`
3. Resume training: `bash scripts/run_training_ddp.sh`
4. Training will auto-resume from latest checkpoint

### 9. Download Results

After training completes:

```bash
# From your local machine
scp -i ~/.ssh/dit-training-key.pem -r \
  ubuntu@<instance-ip>:~/msml_612_project/outputs \
  ~/Downloads/training_results/

scp -i ~/.ssh/dit-training-key.pem -r \
  ubuntu@<instance-ip>:~/msml_612_project/model \
  ~/Downloads/training_results/
```

### 10. Terminate Instance

**IMPORTANT:** Always terminate when done to avoid charges!

```bash
# From your local machine
aws ec2 cancel-spot-instance-requests --spot-instance-request-ids <sir-xxxxx>
aws ec2 terminate-instances --instance-ids <i-xxxxx>
```

Or via AWS Console:

1. Go to EC2 Dashboard
2. Select your instance
3. Actions → Instance State → Terminate

## Cost Monitoring

### Set Up Billing Alerts

1. Go to AWS Billing Dashboard
2. Budgets → Create Budget
3. Set alerts at $50 and $80

### Check Current Costs

```bash
# View current month's costs
aws ce get-cost-and-usage \
  --time-period Start=2025-12-01,End=2025-12-31 \
  --granularity MONTHLY \
  --metrics BlendedCost
```

Or check in AWS Console: Billing Dashboard → Cost Explorer

## Troubleshooting

### GPU Not Detected

```bash
# Verify CUDA
nvcc --version

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory Error

- Reduce batch size in `config.yaml` (try 32 instead of 64)
- Enable gradient checkpointing (if implemented)

### Slow Training

- Verify all 4 GPUs are being used: `watch nvidia-smi`
- Check DDP is working: Look for "Using DDP with 4 GPUs" in log

### Checkpoint Issues

- Check `model/` directory exists: `ls -la model/`
- Verify permissions: `chmod -R 755 model/`
- Check disk space: `df -h`

## Tips for Maximizing Your $100

1. **Always use Spot instances** - 70% cost savings
2. **Enable auto_resume** - Avoid wasting progress on interruptions
3. **Use a smaller test run first** - Set `num_epochs: 2` to verify everything works
4. **Monitor costs daily** - Check AWS Billing Dashboard
5. **Terminate immediately after training** - Don't leave instances running idle
6. **Use screen/tmux** - Keep training running if SSH disconnects

```bash
# Install and use screen
sudo apt-get install screen
screen -S training
bash scripts/run_training_ddp.sh
# Press Ctrl+A then D to detach
# Reconnect with: screen -r training
```

## Expected Timeline

- **Setup:** ~10-15 minutes
- **Training (200 epochs):** ~4-6 hours
- **Total cost:** ~$5-7 (Spot pricing)
- **Remaining budget:** ~$93-95 for more experiments!

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs in `training.log`
3. Check AWS CloudWatch for instance logs
4. Ensure security groups allow SSH access
