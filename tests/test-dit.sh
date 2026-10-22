#!/bin/bash

set -eo pipefail

for backend in CUDA0 CPU; do
    ./test-dit.py --device $backend 2>&1 | tee ${backend}-dit.log
done
