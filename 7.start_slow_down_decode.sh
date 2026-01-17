#!/bin/bash

echo "will send slow_down request to DECODE_HEAD_NODE($DECODE_HEAD_NODE)"
curl -H "Content-Type: application/json" \
    -d "{\"forward_sleep_time\": 60}" \
    -X POST "http://$DECODE_HEAD_NODE:30000/slow_down"

echo "slow_down request sent successfully"

