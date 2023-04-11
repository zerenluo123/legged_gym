#!/bin/bash
CACHE=$1
python play.py --task=go1 --s_flag=1 \
--output_name=ppo/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy