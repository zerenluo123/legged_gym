#!/bin/bash
CACHE=$1
python play.py --task=go1 --s_flag=2 \
--algo=ProprioAdapt \
--priv_info --proprio_adapt \
--output_name=go1/"${CACHE}" \
--checkpoint_model=last.pt \
--export_policy