#!/bin/bash
set -euxo pipefail

echo "will start nsys profile"
echo "-----please make sure you have uncomment the nsys command in launch_server_in_docker_fp4.sh-----"
echo "make sure you are on the DECODE_HEAD_NODE($DECODE_HEAD_NODE) and run this script"
export WORKSPACE=$(realpath "$PWD")
mkdir -p ${WORKSPACE}/../logs
export LOG_FILENAME=${WORKSPACE}/../logs/sglang_NODE_${SLURM_NODEID}_$(date +"%Y%m%d_%H%M%S")

nsys start --backtrace none --sample none --cpuctxsw none --force-overwrite true --output ${LOG_FILENAME}
sleep 0.5
nsys stop
nsys stats --force-export=true ${LOG_FILENAME}.nsys-rep 2>&1 |tee ${LOG_FILENAME}.txt
