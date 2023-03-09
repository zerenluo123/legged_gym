# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
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

from hora.algo.ppo.ppo import PPO
from hora.algo.padapt.padapt import ProprioAdapt
from hora.tasks import isaacgym_task_map
from hora.utils.reformat import omegaconf_to_dict, print_dict
from hora.utils.misc import set_np_formatting, set_seed, git_hash, git_diff_config

import numpy as np
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from shutil import copyfile

from rsl_rl.runners import OnPolicyRunner

def train(args):
    # set numpy formatting for printing only
    set_np_formatting()

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


# ## OmegaConf & Hydra Config
#
# # Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
# OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
# OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
# OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# # allows us to resolve default arguments which are copied in multiple places in the config.
# # used primarily for num_ensv
# OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)
#
#
# @hydra.main(config_name='config', config_path='configs')
# def main(config: DictConfig):
#     if config.checkpoint:
#         config.checkpoint = to_absolute_path(config.checkpoint)
#
#     # set numpy formatting for printing only
#     set_np_formatting()
#
#     # sets seed. if seed is -1 will pick a random one
#     config.seed = set_seed(config.seed)
#
#     cprint('Start Building the Environment', 'green', attrs=['bold'])
#     env = isaacgym_task_map[config.task_name](
#         config=omegaconf_to_dict(config.task),
#         sim_device=config.sim_device,
#         graphics_device_id=config.graphics_device_id,
#         headless=config.headless,
#     )
#
#     output_dif = os.path.join('outputs', config.train.ppo.output_name)
#     os.makedirs(output_dif, exist_ok=True)
#     agent = eval(config.train.algo)(env, output_dif, full_config=config)
#     if config.test:
#         agent.restore_test(config.train.load_path)
#         agent.test()
#     else:
#         date = str(datetime.datetime.now().strftime('%m%d%H'))
#         print(git_diff_config('./'))
#         os.system(f'git diff HEAD > {output_dif}/gitdiff.patch')
#         with open(os.path.join(output_dif, f'config_{date}_{git_hash()}.yaml'), 'w') as f:
#             f.write(OmegaConf.to_yaml(config))
#
#         # check whether execute train by mistake:
#         best_ckpt_path = os.path.join(
#             'outputs', config.train.ppo.output_name,
#             'stage1_nn' if config.train.algo == 'PPO' else 'stage2_nn', 'best.pth'
#         )
#         if os.path.exists(best_ckpt_path):
#             user_input = input(
#                 f'are you intentionally going to overwrite files in {config.train.ppo.output_name}, type yes to continue \n')
#             if user_input != 'yes':
#                 exit()
#
#         agent.restore_train(config.train.load_path)
#         agent.train()
#
#
# if __name__ == '__main__':
#     main()
