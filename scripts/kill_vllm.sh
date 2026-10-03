#!/bin/bash
# Kill orphaned vLLM engine processes (they keep GPU memory after the parent exits).
for p in $(ps -eo pid,comm | awk '$2 ~ /^VLLM::/ {print $1}'); do kill -9 $p; done
