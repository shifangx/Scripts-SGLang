#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
export WORKSPACE=$(realpath "$PWD")
mkdir -p ${WORKSPACE}/../logs

source ${WORKSPACE}/../env/node_list_env.sh
export SGLANG_DOCKER_IMAGE=${WORKSPACE}/../docker/sglang:nightly-dev-20251121-c56fc424.sqsh

export ENROOT_MOUNT_HOME=no

echo "DECODE_NODE_LIST=$DECODE_NODE_LIST"
echo "DECODE_HEAD_NODE=$DECODE_HEAD_NODE"
srun --ntasks=12 --nodes=12 --nodelist=${DECODE_NODE_LIST} \
--container-image=${SGLANG_DOCKER_IMAGE} \
--container-mounts /lustre:/lustre \
--container-workdir="$PWD" \
bash ${WORKSPACE}/launch_server_in_docker_fp4.sh decode 

