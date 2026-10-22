#!/bin/bash

set -eu

# Multi-GPU: add --device <name> below (CUDA0, CUDA1, Vulkan0, CPU)

./build/mm-server \
    --host 0.0.0.0 \
    --port 8086 \
    --models ./models \
    --max-batch 1
