#!/bin/bash

echo "launch router in docker"
echo "make sure you are on the PREFILL_HEAD_NODE($PREFILL_HEAD_NODE) and run this script"
echo "PREFILL_HEAD_NODE=$PREFILL_HEAD_NODE"
echo "DECODE_HEAD_NODE=$DECODE_HEAD_NODE"
python -m sglang_router.launch_router \
    --mini-lb \
    --pd-disaggregation \
    --prefill http://$PREFILL_HEAD_NODE:30000 8991 \
    --decode http://$DECODE_HEAD_NODE:30000  \
    --host 0.0.0.0 \
    --port 7000
