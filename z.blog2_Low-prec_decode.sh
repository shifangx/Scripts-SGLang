#!/bin/bash

# This script is copied from [Instructions for DeepSeek on GB200 #10903](https://github.com/sgl-project/sglang/issues/10903)
# It shows how to benchmark decode server with low-precision computation.
# This script is not used in this test directory, just for reference.

export SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1
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
export FLASHINFER_AUTOTUNER_LOAD_FROM_FILE=1


if [[ "$1" == "prefill" ]]; then
	echo "start prefill server in docker"

  /data/numa0/tom/venvs/sgl/bin/python3 -m sglang.launch_server \
    --disaggregation-mode prefill \
    --host 0.0.0.0 \
    --decode-log-interval 1 \
    --max-running-requests 5632 \
    --context-length 2176 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --watchdog-timeout 1000000 \
    --disable-chunked-prefix-cache \
    --attention-backend trtllm_mla \
    --kv-cache-dtype fp8_e4m3 \
    --enable-single-batch-overlap \
    --tp-size 4 \
    --dp-size 4 \
    --enable-dp-attention \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --chunked-prefill-size 65536 \
    --eplb-algorithm deepseek \
    --model-path /data/numa0/tom/downloaded_models/models--nvidia--DeepSeek-R1-0528-FP4-v2/snapshots/25a138f28f49022958b9f2d205f9b7de0cdb6e18 \
    --trust-remote-code \
    --disable-cuda-graph \
    --port 30000 \
    --mem-fraction-static 0.84 \
    --max-total-tokens 131072 \
    --max-prefill-tokens 16384 \
    --load-balance-method round_robin \
    --quantization modelopt_fp4 \
    --enable-ep-moe \
    --moe-runner-backend flashinfer_cutlass \
    --disaggregation-bootstrap-port 8991

elif [[ "$1" == "decode" ]]; then
	echo "start decode server in docker"
  export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1408
  export SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH=1
  export SGLANG_FP4_GEMM_BACKEND=cutlass

  /data/numa0/tom/venvs/sgl/bin/python3 -m sglang.launch_server \
    --disaggregation-mode decode \
    --host 0.0.0.0 \
    --decode-log-interval 1 \
    --max-running-requests 67584 \
    --context-length 2176 \
    --disable-radix-cache \
    --disable-shared-experts-fusion \
    --watchdog-timeout 1000000 \
    --disable-chunked-prefix-cache \
    --attention-backend trtllm_mla \
    --kv-cache-dtype fp8_e4m3 \
    --dist-init-addr 192.168.3.12:5757 \
    --nnodes 12 \
    --node-rank 0 \
    --init-expert-location /data/numa0/tom/primary_synced/adhoc_local2server/expert_distribution_recorder_1758079614.3163116.pt \
    --enable-single-batch-overlap \
    --model-path /data/numa0/tom/downloaded_models/models--nvidia--DeepSeek-R1-0528-FP4-v2/snapshots/25a138f28f49022958b9f2d205f9b7de0cdb6e18 \
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
    --cuda-graph-bs 1408 \
    --num-reserved-decode-tokens 112 \
    --ep-num-redundant-experts 32 \
    --eplb-algorithm deepseek \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --prefill-round-robin-balance \
    --max-total-tokens 3122380 \
    --quantization modelopt_fp4 \
    --moe-runner-backend flashinfer_cutedsl

else
	echo "invalid argument"
	echo "example usage: ./launch_server.sh [prefill|decode]"
	exit 1
fi

