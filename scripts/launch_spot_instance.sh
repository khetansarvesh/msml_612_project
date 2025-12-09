#!/bin/bash
# Launch g4dn.12xlarge Spot Instance for DDP Training
# Usage: bash scripts/launch_spot_instance.sh

set -e

echo "🚀 Launching g4dn.12xlarge Spot Instance..."

# Configuration
INSTANCE_TYPE="g4dn.12xlarge"
AMI_ID="ami-0c55b159cbfafe1f0"  # Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 20.04) - update for your region
KEY_NAME="dit-training-key"
MAX_PRICE="1.50"  # Max price per hour (current spot ~$1.17/hr)
REGION="us-east-1"
VOLUME_SIZE=100  # GB

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Configuration:${NC}"
echo "  Instance Type: $INSTANCE_TYPE"
echo "  Max Price: \$$MAX_PRICE/hour"
echo "  Region: $REGION"
echo "  Storage: ${VOLUME_SIZE}GB"
echo ""

# Check if key pair exists
echo "Checking for SSH key pair..."
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    echo -e "${YELLOW}Key pair not found. Creating new key pair...${NC}"
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text \
        --region "$REGION" > ~/.ssh/${KEY_NAME}.pem
    chmod 400 ~/.ssh/${KEY_NAME}.pem
    echo -e "${GREEN}✓ Key pair created and saved to ~/.ssh/${KEY_NAME}.pem${NC}"
else
    echo -e "${GREEN}✓ Key pair already exists${NC}"
fi

# Create or get security group
echo "Setting up security group..."
SG_NAME="dit-training-sg"
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ]; then
    echo "Creating new security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Security group for DIT training on AWS" \
        --region "$REGION" \
        --query 'GroupId' \
        --output text)
    
    # Allow SSH from anywhere (you can restrict this to your IP)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region "$REGION"
    
    echo -e "${GREEN}✓ Security group created: $SG_ID${NC}"
else
    echo -e "${GREEN}✓ Using existing security group: $SG_ID${NC}"
fi

# Get latest Deep Learning AMI
echo "Finding latest Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=Deep Learning AMI GPU PyTorch * (Ubuntu 20.04)*" \
              "Name=state,Values=available" \
    --region "$REGION" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

echo -e "${GREEN}✓ Using AMI: $AMI_ID${NC}"

# Create launch specification file
cat > /tmp/spot-launch-spec.json <<EOF
{
  "ImageId": "$AMI_ID",
  "InstanceType": "$INSTANCE_TYPE",
  "KeyName": "$KEY_NAME",
  "SecurityGroupIds": ["$SG_ID"],
  "BlockDeviceMappings": [
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "VolumeSize": $VOLUME_SIZE,
        "VolumeType": "gp3",
        "DeleteOnTermination": true
      }
    }
  ],
  "TagSpecifications": [
    {
      "ResourceType": "instance",
      "Tags": [
        {
          "Key": "Name",
          "Value": "DIT-Training-4GPU"
        },
        {
          "Key": "Project",
          "Value": "MSML-612"
        }
      ]
    }
  ]
}
EOF

# Request spot instance
echo "Requesting spot instance..."
SPOT_REQUEST=$(aws ec2 request-spot-instances \
    --spot-price "$MAX_PRICE" \
    --instance-count 1 \
    --type "persistent" \
    --launch-specification file:///tmp/spot-launch-spec.json \
    --region "$REGION" \
    --output json)

SPOT_REQUEST_ID=$(echo "$SPOT_REQUEST" | jq -r '.SpotInstanceRequests[0].SpotInstanceRequestId')

echo -e "${GREEN}✓ Spot request created: $SPOT_REQUEST_ID${NC}"
echo "Waiting for instance to launch..."

# Wait for spot request to be fulfilled
sleep 5
INSTANCE_ID=""
for i in {1..30}; do
    INSTANCE_ID=$(aws ec2 describe-spot-instance-requests \
        --spot-instance-request-ids "$SPOT_REQUEST_ID" \
        --region "$REGION" \
        --query 'SpotInstanceRequests[0].InstanceId' \
        --output text 2>/dev/null || echo "None")
    
    if [ "$INSTANCE_ID" != "None" ] && [ -n "$INSTANCE_ID" ]; then
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 10
done

if [ "$INSTANCE_ID" == "None" ] || [ -z "$INSTANCE_ID" ]; then
    echo -e "${YELLOW}⚠ Instance not yet assigned. Check spot request status:${NC}"
    echo "  aws ec2 describe-spot-instance-requests --spot-instance-request-ids $SPOT_REQUEST_ID --region $REGION"
    exit 1
fi

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
echo "  Spot Request ID: $SPOT_REQUEST_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  Instance Type: $INSTANCE_TYPE (4x NVIDIA T4)"
echo ""
echo -e "${YELLOW}Connect with:${NC}"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Wait 2-3 minutes for instance to fully initialize"
echo "  2. SSH into the instance (command above)"
echo "  3. Run: bash scripts/setup_aws_instance.sh"
echo ""
echo -e "${YELLOW}To Terminate:${NC}"
echo "  aws ec2 cancel-spot-instance-requests --spot-instance-request-ids $SPOT_REQUEST_ID --region $REGION"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""

# Save instance info
echo "$INSTANCE_ID" > /tmp/dit_instance_id.txt
echo "$PUBLIC_IP" > /tmp/dit_instance_ip.txt
echo "$SPOT_REQUEST_ID" > /tmp/dit_spot_request_id.txt

echo -e "${GREEN}Instance info saved to /tmp/dit_instance_*.txt${NC}"
