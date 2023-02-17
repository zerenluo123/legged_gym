# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, scenarios

import numpy as np
import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners.meta_strategy_optimization.group_up_mso_optimizer import GroupUPMSOOptimizer

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_limb_mass = False
    env_cfg.domain_rand.randomize_motor_strength = False

    env_cfg.env.episode_length_s = 1e7 # no termination when reach max length

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # set hand-crafted environment parameters
    set_env_params(env_cfg, env)

    # set evaluation scenario
    scenario = scenarios.HorizontalLine(env, avg_vel=0.4, length=10)

    # load policy
    train_cfg.runner.resume = True  # set the mode to be evaluation
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 5  # which joint is used for logging
    stop_state_log = 2000  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    env.reset()
    obs_dict = env.get_observations()
    obs = obs_dict['obs'].to(env.device)

    # evaluation of optimum
    scenario.reset()
    for i in range(10 * int(env.max_episode_length)):
        command_vec = scenario.advance()

        actions = policy(obs.detach())
        obs_dict, rews, dones, infos = env.step(actions.detach())
        obs = obs_dict['obs']
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                        'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale +
                                      env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )

        elif i == stop_state_log:
            logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


def set_env_params(env_cfg, env):
    for i in range(env.num_envs):

        # find env instance
        env_curr = env.envs[i]
        handle = env.gym.find_actor_handle(env_curr, env.cfg.asset.name)

        # MOTOR STRENGTH ~ 24 dims
        # w_stiffness = {'FR_hip_joint': 0.8, 'FR_thigh_joint': 0.8, 'FR_calf_joint': 0.8}  # percentage
        # w_damping = {'FR_hip_joint': 0.8, 'FR_thigh_joint': 0.8, 'FR_calf_joint': 0.8}  # percentage
        w_stiffness = {'hip_joint': 0.1, 'thigh_joint': 0.1, 'calf_joint': 0.1}  # percentage
        w_damping = {'hip_joint': 0.1, 'thigh_joint': 0.1, 'calf_joint': 0.1}  # percentage

        dof_props = env.gym.get_actor_dof_properties(env_curr, handle)
        # ! set gym's PD controller
        for i in range(env.num_dof):
            name = env.dof_names[i]
            for dof_name in w_stiffness.keys():
                if dof_name in name:
                    dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                    dof_props['stiffness'][i] *= (1 - w_stiffness[dof_name])  # env.Kp
                    dof_props['damping'][i] *= (1 - w_damping[dof_name])  # env.Kd
        print("p gain ", dof_props['stiffness'])
        # print("d gain ", dof_props['damping'])
        env.gym.set_actor_dof_properties(env_curr, handle, dof_props)

        # filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'go1', 'exported',
        #                         'frames', f"{i}.png")
        # env.gym.write_viewer_image_to_file(env.viewer, filename)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
