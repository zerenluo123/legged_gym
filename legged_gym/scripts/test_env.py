from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from isaacgym import gymapi

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

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # TODO: hand-crafted policy

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 300  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    # Look at the first env
    cam_pos = gymapi.Vec3(-0.8, -1.0, 0.7)
    cam_target = gymapi.Vec3(0.5, 0.0, 0)
    env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)

    for i in range(10 * int(env.max_episode_length)):
        # Sets the pose to be fixed at origin. equivalent to hang the robot
        fixed_pose = torch.tensor([[0.0, 0.0, 0.2, 0., -0., -0., 0.5]], dtype=torch.float)
        env._set_body_pose_to_actors_fixed_at_origin(fixed_pose)

        osc1 = 0.5 * np.sin(i / 20)  # sinusoidal function
        # osc1 = 1.333 / (1 + np.exp(10*(-i+150))) # approximated spike functin(sigmoid)
        actions = torch.tensor([[0, osc1, 0,
                                 0, osc1, 0,
                                 0, osc1, 0,
                                 0, osc1, 0]], dtype=torch.float)
        obs, _, rews, dones, infos = env.step(actions.detach())

        if i < stop_state_log:
            logger.log_states(
                {
                    # 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[0, joint_index].item(),
                    # 'dof_pos_target': env.target_poses[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_actions': env.actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[0, joint_index].item(),
                    # 'act_pos': env.act_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                }
            )
        elif i == stop_state_log:
            logger.plot_states()
            logger.print_rewards()


if __name__ == '__main__':
    args = get_args()
    play(args)
