#!/bin/bash
# terminate the instance, wait, delete the SG and key pair; then describe to verify
S=/private/tmp/claude-501/-Users-mei-git-av1gpu/3ff16727-2f42-41f2-b891-0efc6c7767ab/scratchpad/lr08; R=us-east-1; N=av1gpu-lr08
. $S/ids.txt 2>/dev/null
for i in $(aws ec2 describe-instances --region $R --filters "Name=tag:Name,Values=$N" "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" --query 'Reservations[].Instances[].InstanceId' --output text); do
  aws ec2 terminate-instances --region $R --instance-ids $i >/dev/null; aws ec2 wait instance-terminated --region $R --instance-ids $i; echo "terminated $i"; done
for g in $(aws ec2 describe-security-groups --region $R --filters "Name=group-name,Values=$N" --query 'SecurityGroups[].GroupId' --output text); do
  aws ec2 delete-security-group --region $R --group-id $g && echo "deleted SG $g"; done
aws ec2 describe-key-pairs --region $R --key-names $N >/dev/null 2>&1 && aws ec2 delete-key-pair --region $R --key-name $N && echo "deleted key $N"
echo "--- verify"
aws ec2 describe-instances --region $R --filters "Name=tag:Name,Values=$N" --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text
aws ec2 describe-security-groups --region $R --filters "Name=group-name,Values=$N" --query 'SecurityGroups[].GroupId' --output text
aws ec2 describe-key-pairs --region $R --filters "Name=key-name,Values=$N" --query 'KeyPairs[].KeyName' --output text
aws ec2 describe-volumes --region $R --filters "Name=tag:Name,Values=$N" --query 'Volumes[].[VolumeId,State]' --output text
rm -f $S/key.pem
echo TEARDOWN_DONE
