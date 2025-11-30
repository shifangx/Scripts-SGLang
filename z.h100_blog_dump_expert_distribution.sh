#!/bin/bash

# This script is copied from [Instruction for Running DeepSeek with Large-scale PD and EP #6017](https://github.com/sgl-project/sglang/issues/6017)
# It shows how to dump expert distribution record from server.
# This script is not used in this test directory, just for reference.

# prefill
SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=4 \
MC_TE_METRIC=true \
SGLANG_TORCH_PROFILER_DIR=/host_home/temp_sglang_server2local \
SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/host_home/temp_sglang_server2local \
SGLANG_TBO_DEBUG=1 \
PYTHONUNBUFFERED=1 \
python3 -m sglang.launch_server \
  --model-path /dev/shm/DeepSeek-V3-0324 \
  --disaggregation-ib-device mlx5_1 \
  --disaggregation-mode prefill \
  --dist-init-addr 10.5.55.1:5757 \
  --nnodes 4 \
  --node-rank 0 \
  --tp-size 32 \
  --dp-size 32 \
  --enable-dp-attention \
  --decode-log-interval 1 \
  --enable-deepep-moe \
  --page-size 1 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --disable-radix-cache \
  --watchdog-timeout 1000000 \
  --enable-two-batch-overlap \
  --expert-distribution-recorder-mode stat \
  --disable-overlap-schedule \
  --expert-distribution-recorder-buffer-size -1 \
  --deepep-mode normal \
  --mem-fraction-static 0.82 \
  --chunked-prefill-size 524288 \
  --max-running-requests 8192 \
  --max-total-tokens 131072 \
  --context-length 8192 \
  --ep-num-redundant-experts 32 \
  --ep-dispatch-algorithm dynamic \
  --eplb-algorithm deepseek \
  --deepep-config /host_home/primary_synced/tom_sglang_server/misc/deepep_vp.json

# decode
MC_TE_METRIC=true \
SGLANG_TORCH_PROFILER_DIR=/host_home/temp_sglang_server2local \
SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/host_home/temp_sglang_server2local \
SGLANG_TBO_DEBUG=1 \
PYTHONUNBUFFERED=1 \
python3 -m sglang.launch_server \
  --model-path /dev/shm/DeepSeek-V3-0324 \
  --disaggregation-ib-device mlx5_1 \
  --disaggregation-mode decode \
  --dist-init-addr 10.5.55.5:5757 \
  --nnodes 9 \
  --node-rank 0 \
  --tp-size 72 \
  --dp-size 72 \
  --enable-dp-attention \
  --decode-log-interval 1 \
  --enable-deepep-moe \
  --page-size 1 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --disable-radix-cache \
  --watchdog-timeout 1000000 \
  --enable-two-batch-overlap \
  --expert-distribution-recorder-mode stat \
  --disable-overlap-schedule \
  --expert-distribution-recorder-buffer-size -1 \
  --deepep-mode low_latency \
  --mem-fraction-static 0.81 \
  --max-running-requests 18432 \
  --context-length 4500 \
  --ep-num-redundant-experts 32 \
  --cuda-graph-bs 256 \
  --num-reserved-decode-tokens YOUR_VALUE

curl -X POST -H 'Content-Type: application/json' 'http://10.5.55.1:30000/start_expert_distribution_record' -d '{}' 
curl -X POST -H 'Content-Type: application/json' 'http://10.5.55.5:30000/start_expert_distribution_record' -d '{}' 
curl -X POST -H 'Content-Type: application/json' 'http://10.5.55.5:30000/slow_down' -d '{"forward_sleep_time": 90.0}' 
python3 -m sglang.bench_one_batch_server \
  --base-url http://10.5.55.1:8000 \
  --model-path /dev/shm/DeepSeek-V3-0324 \
  --batch-size 40000 \
  --input-len 2000 \
  --output-len 100 \
  --skip-warmup
# after a while
curl -X POST -H 'Content-Type: application/json' 'http://10.5.55.5:30000/slow_down' -d '{"forward_sleep_time": null}' 
# after a while
curl -X POST -H 'Content-Type: application/json' 'http://10.5.55.1:30000/dump_expert_distribution_record' -d '{}' 
curl -X POST -H 'Content-Type: application/json' 'http://10.5.55.5:30000/dump_expert_distribution_record' -d '{}' 
