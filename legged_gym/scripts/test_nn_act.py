import glob
import json

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from isaacgym import gymapi

class KinematicActor:
    def __init__(self):
        # Flag to inidicate whether the object has currently loaded a reference
        # motion to reproduce.
        self.ref_motion_is_loaded = False
        # Stores the frames of the reference motion
        self.ref_motion_frames = None

    def load_reference_motion(self, dof_frames):
        """ Loads the reference motion from `dof_frames`. A reference motion
        consists of a sequence of frames that contains the root position (3)
        and orientation (4) of the character + the PD targets for each joint.

        """

        self.ref_motion_is_loaded = True
        self.ref_motion_frames = dof_frames

        self.motion_i = 0
        self.motion_length = self.ref_motion_frames.shape[0]

    def load_next_frame(self):
        """ Reproduces the next frame of the reference motion. """

        # Load current frame
        frame = self.ref_motion_frames[self.motion_i, :]

        # This attribute is used externally by `set_body_pose_actors`.
        # `pose_target` for this actor is set using the tensor API (which sets
        # the value for ALL actors simultaneously).
        self.actions = frame

        # Go to next frame
        self.motion_i += 1
        # Loop around the motion
        self.motion_i = self.motion_i % self.motion_length

        if self.motion_i == 0:
            print("finish episode")

    def load_next_frame_indexed(self, idx):
        """ Reproduces the next "idx"-th frame of the real motion. """
        # Load current frame
        frame = self.ref_motion_frames[idx * self.motion_i, :]

        self.real_frame = frame
        # Go to next frame
        self.motion_i += 1
        # Loop around the motion
        self.motion_i = self.motion_i % self.motion_length



def play(args):
    # Load the reference motions from .txt files
    motion_txts = glob.glob("actions/*.txt")
    print(f"Loaded reference motions: `{motion_txts}`")
    motion_frames = [np.loadtxt(fn) for fn in motion_txts]

    # Create actor-hand-crafted policy
    kin_actor = KinematicActor()
    rand_i = 1  # Fixing a motion
    # kin_actor.load_reference_motion(motion_frames[rand_i])

    # Load the real motions from .txt files (different instantiations)
    real_pos_txts = glob.glob("motions/*.txt")
    print(f"Loaded real pose: `{real_pos_txts}`")
    real_pos_frames = [np.loadtxt(fn) for fn in real_pos_txts]
    real_vel_txts = glob.glob("velocities/*.txt")
    print(f"Loaded real velocity: `{real_vel_txts}`")
    real_vel_frames = [np.loadtxt(fn) for fn in real_vel_txts]
    real_act_txts = glob.glob("acts/*.txt")
    print(f"Loaded real actions: `{real_act_txts}`")
    real_act_frames = [np.loadtxt(fn) for fn in real_act_txts]

    # Create real pos logger
    real_pos_actor = KinematicActor()
    real_pos_actor.load_reference_motion(real_pos_frames[rand_i])
    real_vel_actor = KinematicActor()
    real_vel_actor.load_reference_motion(real_vel_frames[rand_i])
    real_act_actor = KinematicActor()
    real_act_actor.load_reference_motion(real_act_frames[rand_i])

    # env
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

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 500 # number of steps before plotting states
    # Look at the first env
    cam_pos = gymapi.Vec3(-0.8, -1.0, 0.7)
    cam_target = gymapi.Vec3(0.5, 0.0, 0)
    env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)

    for i in range(10 * int(env.max_episode_length)):
        # Sets the pose to be fixed at origin. equivalent to hang the robot
        fixed_pose = torch.tensor([[0.0, 0.0, 0.3, 0., -0., -0., 0.5]], dtype=torch.float)
        env._set_body_pose_to_actors_fixed_at_origin(fixed_pose)

        # the action should be the "high-level" measured one
        # kin_actor.load_next_frame()
        # actions = torch.tensor(kin_actor.actions, dtype=torch.float).unsqueeze(0)
        real_act_actor.load_next_frame_indexed(8)
        target_poses = (real_act_actor.real_frame - env.default_dof_pos.cpu().numpy().squeeze()) / env.cfg.control.action_scale
        actions = torch.tensor(target_poses, dtype=torch.float).unsqueeze(0)

        # Note: this action are used to simulate and act-Net in 200 Hz/0.005s
        _, _, rews, dones, infos = env.step(actions.detach())

        # ! for logging: load the real pos for comparison.
        #  Note: the pos measured on the real robot ~ 400 Hz/0.0025, while action is updated in 0.005*4=0.02s~50Hz. So we should take 1 point every 8 points
        real_pos_actor.load_next_frame_indexed(8)
        real_vel_actor.load_next_frame_indexed(8)
        # real_act_actor.load_next_frame_indexed(8)
        real_pos = torch.tensor(real_pos_actor.real_frame, dtype=torch.float).unsqueeze(0)
        real_vel = torch.tensor(real_vel_actor.real_frame, dtype=torch.float).unsqueeze(0)
        real_act = torch.tensor(real_act_actor.real_frame, dtype=torch.float).unsqueeze(0)

        if 5 < i < stop_state_log:
            logger.log_states(
                {
                    # 'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    # TODO: the pos measured on the real robot ~ 400 Hz/0.0025
                    'real_pos': real_pos[robot_index, joint_index].item(),
                    'real_actions': real_act[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    # 'dof_actions': env.actions[robot_index, joint_index].item(),
                    'real_vel': real_vel[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
            logger.print_rewards()






if __name__ == '__main__':
    args = get_args()
    play(args)
