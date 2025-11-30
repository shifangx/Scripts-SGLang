#!/bin/bash

# This script is copied from [Instructions for DeepSeek on GB200 #10903](https://github.com/sgl-project/sglang/issues/10903)
# It shows how to benchmark decode server with high-precision computation.
# This script is not used in this test directory, just for reference.

export SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1
export FLASHINFER_WORKSPACE_BASE=/data/numa0/tom/flashinfer_workspace_base
export SGL_JIT_DEEPGEMM_PRECOMPILE=0
export MC_TE_METRIC=true
export SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000
export SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True
export SGLANG_LOCAL_IP_NIC=eth0
export SGLANG_DUMPER_DIR=/data/numa0/tom/temp
export SGLANG_TORCH_PROFILER_DIR=/data/numa0/tom/temp_sglang_server2local
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/data/numa0/tom/temp_sglang_server2local
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
export SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
export PYTHONUNBUFFERED=1


if [[ "$1" == "prefill" ]]; then
  echo "start prefill server in docker"
  /data/numa0/tom/venvs/sgl/bin/python3 -m sglang.launch_server \
    --disaggregation-mode prefill \
    --host 0.0.0.0 \
    --decode-log-interval 1 \
    --max-running-requests 6144 \
    --context-length 2176 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --watchdog-timeout 1000000 \
    --attention-backend fa4 \
    --page-size 64 \
    --dist-init-addr 192.168.3.228:5757 \
    --nnodes 2 \
    --node-rank 0 \
    --init-expert-location /data/numa0/tom/primary_synced/adhoc_local2server/expert_distribution_recorder_1758079621.5408838.pt \
    --tp-size 8 \
    --dp-size 8 \
    --enable-dp-attention \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --chunked-prefill-size 131072 \
    --eplb-algorithm deepseek \
    --ep-num-redundant-experts 32 \
    --model-path /data/numa0/tom/downloaded_models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52/ \
    --trust-remote-code \
    --disable-cuda-graph \
    --port 30000 \
    --mem-fraction-static 0.84 \
    --max-total-tokens 131072 \
    --max-prefill-tokens 16384 \
    --load-balance-method round_robin \
    --enable-deepep-moe \
    --deepep-mode normal \
    --deepep-config /data/numa0/tom/primary_synced/tom_sglang_server/misc/deepep_a30_ac4853.json \
    --ep-dispatch-algorithm dynamic \
    --disaggregation-bootstrap-port 8991


elif [[ "$1" == "decode" ]]; then
	echo "start decode server in docker"

  export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=768
  /data/numa0/tom/venvs/sgl/bin/python3 -m sglang.launch_server \
    --disaggregation-mode decode \
    --host 0.0.0.0 \
    --decode-log-interval 1 \
    --max-running-requests 36864 \
    --context-length 2176 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --watchdog-timeout 1000000 \
    --disable-chunked-prefix-cache \
    --attention-backend trtllm_mla \
    --dist-init-addr 192.168.3.241:5757 \
    --nnodes 12 \
    --node-rank 0 \
    --init-expert-location /data/numa0/tom/primary_synced/adhoc_local2server/expert_distribution_recorder_1758079614.3163116.pt \
    --model-path /data/numa0/tom/downloaded_models/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52/ \
    --trust-remote-code \
    --tp-size 48 \
    --dp-size 48 \
    --enable-dp-attention \
    --chunked-prefill-size 786432 \
    --mem-fraction-static 0.83 \
    --enable-ep-moe \
    --moe-a2a-backend deepep \
    --deepep-mode low_latency \
    --ep-dispatch-algorithm static \
    --cuda-graph-bs 768 \
    --num-reserved-decode-tokens 176 \
    --ep-num-redundant-experts 32 \
    --eplb-algorithm deepseek \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --prefill-round-robin-balance \
    --max-total-tokens 1838284 \
    --enable-single-batch-overlap

else
	echo "invalid argument"
	echo "example usage: ./launch_server.sh [prefill|decode]"
	exit 1
fi

