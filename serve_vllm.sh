#!/bin/bash

nohup vllm serve Qwen/Qwen2.5-14B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --tensor-parallel-size 8 \
    > output.log 2>&1 &