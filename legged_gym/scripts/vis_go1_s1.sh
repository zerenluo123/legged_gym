#!/bin/bash
CACHE=$1
python train_RMA.py --task=go1 --num_envs=1 --test \
--output_name=go1/"${CACHE}" \
