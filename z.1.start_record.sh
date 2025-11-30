#!/bin/bash

echo "will start expert distribution record"
curl -H "Content-Type: application/json" \
    -d "{}" \
    -X POST "http://$DECODE_HEAD_NODE:30000/start_expert_distribution_record"

echo "expert distribution record started successfully"

