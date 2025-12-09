#!/bin/bash
# Launch g4dn.2xlarge Instance (1x T4 GPU, 8 vCPUs)
# Fits within 8 vCPU limit

set -e

echo "🚀 Launching g4dn.2xlarge Instance (1 GPU)..."

# Configuration
INSTANCE_TYPE="g4dn.2xlarge"
AMI_ID="ami-07e1ee23c621044d8"  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0
KEY_NAME="dit-training-key"
REGION="us-east-1"
VOLUME_SIZE=100  # GB

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Configuration:${NC}"
echo "  Instance Type: $INSTANCE_TYPE"
echo "  GPUs: 1x NVIDIA T4 (16GB)"
echo "  vCPUs: 8 (fits your limit)"
echo "  Pricing: ~\$0.75/hour On-Demand"
echo "  Region: $REGION"
echo ""

# Get existing security group
SG_NAME="dit-training-sg"
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ]; then
    echo -e "${YELLOW}Security group not found. Creating...${NC}"
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Security group for DIT training on AWS" \
        --region "$REGION" \
        --query 'GroupId' \
        --output text)
    
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region "$REGION"
fi

echo -e "${GREEN}✓ Using security group: $SG_ID${NC}"

# Launch instance
echo "Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=DIT-Training-1GPU},{Key=Project,Value=MSML-612}]" \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}✓ Instance launched: $INSTANCE_ID${NC}"

# Wait for instance to be running
echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 Instance Ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Instance Details:${NC}"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  Type: $INSTANCE_TYPE (1x T4, 16GB)"
echo "  Cost: ~\$0.75/hour On-Demand"
echo ""
echo -e "${YELLOW}Budget Estimate:${NC}"
echo "  \$100 = ~133 hours of runtime"
echo "  Training (200 epochs): ~15-20 hours"
echo "  Cost per full run: ~\$11-15"
echo "  Number of runs possible: ~6-9"
echo ""
echo -e "${YELLOW}Connect:${NC}"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Wait 2-3 min for full initialization"
echo "  2. SSH in (command above)"
echo "  3. Clone: git clone <your-repo> msml_612_project"
echo "  4. Setup: cd msml_612_project && bash scripts/setup_aws_instance.sh"
echo "  5. Train: bash scripts/run_training_single_gpu.sh"
echo ""
echo -e "${YELLOW}Terminate (IMPORTANT):${NC}"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo ""

# Save info
echo "$INSTANCE_ID" > /tmp/dit_instance_id.txt
echo "$PUBLIC_IP" > /tmp/dit_instance_ip.txt
echo -e "${GREEN}Instance info saved to /tmp/dit_instance_*.txt${NC}"
