#!/bin/bash
CACHE=$1
python train_RMA.py --task=go1 \
--num_envs=1 --test \
--lin_vel_x=0.5 --lin_vel_y=0.0 --heading=0.0 \
--algo=ProprioAdapt \
--priv_info --proprio_adapt \
--output_name=go1/"${CACHE}" \
--checkpoint_model=outputs/go1/"${CACHE}"/stage2_nn/last.pth \
