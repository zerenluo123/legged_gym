from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import rospy
from sensor_msgs.msg import Joy

class PlayJoy():
    def __init__(self, args):
        self.args = args

        self.joy_cmd_velx = 0.0
        self.joy_cmd_vely = 0.0
        self.joy_cmd_heading = 0.0
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.joy_callback, queue_size=1)

        self.env_cfg, self.train_cfg = task_registry.get_cfgs(name=args.task)
        # override some parameters for testing
        self.env_cfg.env.num_envs = min(self.env_cfg.env.num_envs, 25)
        self.env_cfg.terrain.num_rows = 5
        self.env_cfg.terrain.num_cols = 5
        self.env_cfg.terrain.curriculum = False
        self.env_cfg.noise.add_noise = False
        self.env_cfg.domain_rand.randomize_friction = False
        self.env_cfg.domain_rand.push_robots = False

        # fixed velocity direction evaluation (make sure the value is within the training range)
        self.env_cfg.commands.ranges.lin_vel_x = [self.joy_cmd_velx, self.joy_cmd_velx]
        self.env_cfg.commands.ranges.lin_vel_y = [self.joy_cmd_vely, self.joy_cmd_vely]
        self.env_cfg.commands.ranges.heading = [self.joy_cmd_heading, self.joy_cmd_heading]


    def play(self):
        # prepare environment
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        obs = env.get_observations()
        # load policy
        self.train_cfg.runner.resume = True  # set the mode to be evalution
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=self.args.task, args=self.args, train_cfg=self.train_cfg)
        policy = ppo_runner.get_inference_policy(device=env.device)

        for i in range(10 * int(env.max_episode_length)):
            actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())

            # cmd's linear velocity changing module
            env._change_cmds(self.joy_cmd_velx, self.joy_cmd_vely, self.joy_cmd_heading)


        rospy.spin()


    def joy_callback(self, joy_msg):
        self.joy_cmd_velx = joy_msg.axes[4] * 1.0
        self.joy_cmd_vely = joy_msg.axes[3] * 1.0
        self.joy_cmd_heading = joy_msg.axes[0] * 1.5

    def run(self):
        rospy.spin()



if __name__ == '__main__':
    rospy.init_node('play_policy_with_joy')

    args = get_args()
    play_joy = PlayJoy(args)
    play_joy.play()

    # play_joy.run()


