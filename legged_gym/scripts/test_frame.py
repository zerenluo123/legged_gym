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
        self.pd_targets = frame

        # Go to next frame
        self.motion_i += 1
        # Loop around the motion
        self.motion_i = self.motion_i % self.motion_length

        if self.motion_i == 0:
            print("finish episode")


def play(args):
    # Load the reference motions from .txt files
    motion_txts = glob.glob("motions/*.txt")
    print(f"Loaded reference motions: `{motion_txts}`")
    motion_frames = [np.loadtxt(fn) for fn in motion_txts]


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

    # Create actor # TODO: hand-crafted policy
    kin_actor = KinematicActor()
    rand_i = 1  # Fixing a motion
    kin_actor.load_reference_motion(motion_frames[rand_i])

    # Look at the first env
    cam_pos = gymapi.Vec3(-0.8, -1.0, 0.7)
    cam_target = gymapi.Vec3(0.5, 0.0, 0)
    env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)

    for i in range(10 * int(env.max_episode_length)):
        # Sets the pose to be fixed at origin for every actor
        fixed_pose = torch.tensor([[0.0, 0.0, 0.6, 0., -0., -0., 0.5]], dtype=torch.float)
        env._set_body_pose_to_actors_fixed_at_origin(fixed_pose)

        kin_actor.load_next_frame()
        pd_targets = torch.tensor(kin_actor.pd_targets, dtype=torch.float).unsqueeze(0)

        _, _, rews, dones, infos = env.step(pd_targets.detach())

        # osc1 = 1.0 * np.sin(i / 20)  # sinusoidal function
        # actions = torch.tensor([[0, 0, 0,
        #                          0, 0, 0,
        #                          0, osc1, 0,
        #                          0, osc1, 0]], dtype=torch.float)
        # _, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    args = get_args()
    play(args)
