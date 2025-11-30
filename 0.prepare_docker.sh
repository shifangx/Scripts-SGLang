#!/bin/bash
set -euxo pipefail

export WORKSPACE=$(realpath "$PWD")
mkdir -p ${WORKSPACE}/../docker
export REMOTE_SGLANG_DOCKER_IMAGE=lmsysorg/sglang:nightly-dev-20251121-c56fc424
export SGLANG_DOCKER_IMAGE=${WORKSPACE}/../docker/sglang:nightly-dev-20251121-c56fc424.sqsh
if [[ ! -f ${SGLANG_DOCKER_IMAGE} ]]; then
    echo "download sglang docker image ${REMOTE_SGLANG_DOCKER_IMAGE}"
    enroot import --output ${SGLANG_DOCKER_IMAGE} docker://${REMOTE_SGLANG_DOCKER_IMAGE}
fi
