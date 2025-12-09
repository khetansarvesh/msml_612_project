#!/bin/bash
# Launch g4dn.12xlarge On-Demand Instance for DDP Training
# Use this if spot instance limits are exceeded

set -e

echo "🚀 Launching g4dn.12xlarge On-Demand Instance..."

# Configuration
INSTANCE_TYPE="g4dn.12xlarge"
AMI_ID="ami-07e1ee23c621044d8"  # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0
KEY_NAME="dit-training-key"
REGION="us-east-1"
VOLUME_SIZE=100  # GB

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Configuration:${NC}"
echo "  Instance Type: $INSTANCE_TYPE"
echo "  Pricing: On-Demand (~\$3.91/hour)"
echo "  Region: $REGION"
echo "  Storage: ${VOLUME_SIZE}GB"
echo ""

# Get existing security group
SG_NAME="dit-training-sg"
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ]; then
    echo -e "${YELLOW}Security group not found. Please run launch_spot_instance.sh first to create it.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Using security group: $SG_ID${NC}"

# Launch On-Demand instance
echo "Launching On-Demand instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=DIT-Training-4GPU},{Key=Project,Value=MSML-612}]" \
    --region "$REGION" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${GREEN}✓ Instance launched: $INSTANCE_ID${NC}"

# Wait for instance to be running
echo "Waiting for instance to be in running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 Instance launched successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Instance Details:${NC}"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  Instance Type: $INSTANCE_TYPE (4x NVIDIA T4)"
echo "  Pricing: On-Demand \$3.912/hour"
echo ""
echo -e "${YELLOW}Important - Cost Monitoring:${NC}"
echo "  Your \$100 budget = ~25.5 hours on On-Demand"
echo "  Training time: ~4-6 hours for 200 epochs"
echo "  Estimated cost per run: ~\$15-23"
echo ""
echo -e "${YELLOW}Connect with:${NC}"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Wait 2-3 minutes for instance to fully initialize"
echo "  2. SSH into the instance (command above)"
echo "  3. Clone your repo: git clone <your-repo> msml_612_project"
echo "  4. cd msml_612_project && bash scripts/setup_aws_instance.sh"
echo ""
echo -e "${YELLOW}To Terminate (IMPORTANT):${NC}"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""

# Save instance info
echo "$INSTANCE_ID" > /tmp/dit_instance_id.txt
echo "$PUBLIC_IP" > /tmp/dit_instance_ip.txt

echo -e "${GREEN}Instance info saved to /tmp/dit_instance_*.txt${NC}"
