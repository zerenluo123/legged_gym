# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import isaacgym

import os
import hydra
from datetime import datetime
from termcolor import cprint
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

import numpy as np
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from shutil import copyfile
from legged_gym.utils import Logger

from rsl_rl.runners import OnPolicyRunner

def train(args):
    # if no args passed get command line arguments
    if args is None:
        args = get_args()
    # if config files are passed use them, otherwise load from the name
    # load config files
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    # override cfg from args (if specified)
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)

    cprint('Start Building the Environment', 'green', attrs=['bold'])
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg_dict = class_to_dict(train_cfg)

    output_dif = os.path.join('outputs', args.output_name)
    os.makedirs(output_dif, exist_ok=True)
    RMA_ppo_config = train_cfg_dict['RMA']
    RMA_ppo_config['test'] = True if args.test else False
    agent = eval(args.algo)(env, output_dif, train_config=RMA_ppo_config)

    # save resume path before creating a new log_dir
    if args.test:
        cprint('Start Testing the Policy', 'green', attrs=['bold'])
        # add logger
        agent.logger = Logger(env.dt)
        # load previously trained model
        agent.restore_test(args.checkpoint_model)
        agent.test()
    else:
        cprint('Start Training the Policy', 'green', attrs=['bold'])
        # check whether execute train by mistake:
        best_ckpt_path = os.path.join(
            'outputs', args.output_name,
            'stage1_nn' if args.algo == 'PPO' else 'stage2_nn', 'best.pth'
        )
        if os.path.exists(best_ckpt_path):
            user_input = input(
                f'are you intentionally going to overwrite files in {args.output_name}, type yes to continue \n')
            if user_input != 'yes':
                exit()

        agent.restore_train(args.checkpoint_model) # load trained network from the else where. normally not useful
        agent.train()

if __name__ == '__main__':
    args = get_args()
    train(args)