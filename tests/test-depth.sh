#!/bin/bash

set -eo pipefail

for backend in CUDA0 CPU; do
    ./test-depth.py --device $backend 2>&1 | tee ${backend}-depth.log
done
