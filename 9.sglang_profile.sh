#!/bin/bash

echo "will send start_profile request to DECODE_HEAD_NODE($DECODE_HEAD_NODE)"
curl -X POST -H 'Content-Type: application/json' \
 "http://$DECODE_HEAD_NODE:30000/start_profile" \
 -d '{"num_steps":5}'

echo "start_profile request sent successfully"
