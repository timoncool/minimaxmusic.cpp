#!/bin/bash

set -eo pipefail

for backend in CUDA0 CPU; do
    ./test-cond.py --device $backend 2>&1 | tee ${backend}-cond.log
done
