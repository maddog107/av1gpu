#!/bin/bash
# Launch one CPU instance (c7a/c6a.16xlarge, 64 vCPU, Ubuntu 24.04) for the av1gpu loop-restoration study.
# Spot then on-demand over all subnets.  Everything is tagged Name=av1gpu-lr08;
# SSH only from this IP; root volume DeleteOnTermination; the user data arms a dead-man shutdown (+180 min,
# terminate on shutdown).  Key material goes only to $S/key.pem (never printed).
set -euo pipefail
S=/private/tmp/claude-501/-Users-mei-git-av1gpu/3ff16727-2f42-41f2-b891-0efc6c7767ab/scratchpad/lr08
R=us-east-1; N=av1gpu-lr08; VPC=vpc-3cba1145
T="ResourceType=%s,Tags=[{Key=Name,Value=$N}]"
AMI=$(aws ssm get-parameter --region $R --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id --query Parameter.Value --output text)
MYIP=$(curl -s https://checkip.amazonaws.com)
echo "AMI=$AMI" > $S/ids.txt
umask 077
aws ec2 create-key-pair --region $R --key-name $N --key-type ed25519 --tag-specifications "$(printf $T key-pair)" \
  --query KeyMaterial --output text > $S/key.pem
chmod 600 $S/key.pem
SG=$(aws ec2 create-security-group --region $R --group-name $N --description "av1gpu lr08 SSH" --vpc-id $VPC \
  --tag-specifications "$(printf $T security-group)" --query GroupId --output text)
echo "SG=$SG" >> $S/ids.txt
aws ec2 authorize-security-group-ingress --region $R --group-id $SG --protocol tcp --port 22 --cidr $MYIP/32 >/dev/null
ROOTDEV=$(aws ec2 describe-images --region $R --image-ids $AMI --query 'Images[0].RootDeviceName' --output text)
SPOT='MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}'
UD=$(printf '#!/bin/bash\nshutdown -h +180\n')   # the CLI base64-encodes user data itself
IID=""
for ITYPE in c7a.16xlarge c6a.16xlarge; do
  for MARKET in ${MARKETS:-spot ondemand}; do
    for SUB in subnet-09aac841 subnet-db0da181 subnet-0be91205e7458cd49 subnet-b978cf95 subnet-fb4673c7; do
      EXTRA=(); TAGS=("$(printf $T instance)" "$(printf $T volume)" "$(printf $T network-interface)")
      if [ $MARKET = spot ]; then EXTRA=(--instance-market-options "$SPOT"); TAGS+=("$(printf $T spot-instances-request)"); fi
      if IID=$(aws ec2 run-instances --region $R --image-id $AMI --instance-type $ITYPE --key-name $N \
          --network-interfaces "DeviceIndex=0,SubnetId=$SUB,Groups=$SG,AssociatePublicIpAddress=true" \
          --instance-initiated-shutdown-behavior terminate ${EXTRA[@]+"${EXTRA[@]}"} --user-data "$UD" \
          --block-device-mappings "DeviceName=$ROOTDEV,Ebs={VolumeSize=60,VolumeType=gp3,DeleteOnTermination=true}" \
          --tag-specifications "${TAGS[@]}" --query 'Instances[0].InstanceId' --output text 2>>$S/launch.err); then
        echo "launched $ITYPE $MARKET in $SUB"
        echo "ITYPE=$ITYPE" >> $S/ids.txt; echo "MARKET=$MARKET" >> $S/ids.txt; echo "SUB=$SUB" >> $S/ids.txt
        break 3
      fi
      IID=""
      echo "$ITYPE $MARKET in $SUB failed: $(tail -1 $S/launch.err | cut -c1-160)"
    done
  done
done
[ -n "$IID" ] || { echo "no capacity"; exit 1; }
echo "IID=$IID" >> $S/ids.txt
echo "LAUNCH_EPOCH=$(date +%s)" >> $S/ids.txt
aws ec2 wait instance-running --region $R --instance-ids $IID
IP=$(aws ec2 describe-instances --region $R --instance-ids $IID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "IP=$IP" >> $S/ids.txt
echo "launched; ids in $S/ids.txt"
