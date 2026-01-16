#!/bin/bash
set -euxo pipefail

export WORKSPACE=$(realpath "$PWD")
mkdir -p ${WORKSPACE}/../docker
export REMOTE_SGLANG_DOCKER_IMAGE=lmsysorg/sglang:v0.5.5.post2
export SGLANG_DOCKER_IMAGE=${WORKSPACE}/../docker/sglang:v0.5.5.post2.sqsh
export DEV_SGLANG_DOCKER_IMAGE=${WORKSPACE}/../docker/dev-sglang:v0.5.5.post2.sqsh
if [[ ! -f ${SGLANG_DOCKER_IMAGE} ]]; then
    srun --ntasks=1 --nodes=1 \
    --container-image=${REMOTE_SGLANG_DOCKER_IMAGE} \
    --container-save=${SGLANG_DOCKER_IMAGE} \
    --container-mounts /lustre:/lustre \
    --container-workdir="$PWD" \
    hostname
fi

if [[ ! -f ${DEV_SGLANG_DOCKER_IMAGE} ]]; then
    srun --ntasks=1 --nodes=1 \
    --container-image=${SGLANG_DOCKER_IMAGE} \
    --container-save=${DEV_SGLANG_DOCKER_IMAGE} \
    --container-mounts /lustre:/lustre \
    --container-workdir="$PWD" \
    bash -c "pip install flashinfer-jit-cache==0.5.0 --index-url https://flashinfer.ai/whl/cu129/"
fi
