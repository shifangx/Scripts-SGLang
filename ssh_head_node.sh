#!/bin/bash
set -euxo pipefail
export WORKSPACE=$(realpath "$PWD/")
source "${WORKSPACE}/../env/node_list_env.sh"

ssh -t "${HEAD_NODE}" "cd '${WORKSPACE}' && exec \$SHELL -l"
