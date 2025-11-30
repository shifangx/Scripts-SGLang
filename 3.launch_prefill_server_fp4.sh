#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
export WORKSPACE=$(realpath "$PWD")
mkdir -p ${WORKSPACE}/../logs

source ${WORKSPACE}/../env/node_list_env.sh
export SGLANG_DOCKER_IMAGE=${WORKSPACE}/../docker/sglang:nightly-dev-20251121-c56fc424.sqsh

export ENROOT_MOUNT_HOME=no
echo "PREFILL_NODE_LIST=$PREFILL_NODE_LIST"
echo "PREFILL_HEAD_NODE=$PREFILL_HEAD_NODE"
srun --ntasks=2 --nodes=2 --nodelist=${PREFILL_NODE_LIST} \
--container-image=${SGLANG_DOCKER_IMAGE} \
--container-mounts /lustre:/lustre \
--container-workdir="$PWD" \
bash ${WORKSPACE}/launch_server_in_docker_fp8.sh prefill

