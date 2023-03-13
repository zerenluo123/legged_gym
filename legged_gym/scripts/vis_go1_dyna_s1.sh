#!/bin/bash
CACHE=$1
python train_RMA.py --task=go1 \
--num_envs=1 --test \
--lin_vel_x=0.8 --lin_vel_y=0.0 --heading=0.0 \
--algo=PPO \
--priv_info \
--output_name=go1/"${CACHE}" \
--checkpoint_model=outputs/go1/"${CACHE}"/stage1_nn/last.pth \
--fault=0.9 \
--fault_transitions=0.5 --fault_transitions=0.0 --fault_transitions=0.5