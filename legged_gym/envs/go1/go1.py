from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import nn
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .go1_config import Go1RoughCfg

LEG_NUM = 4
LEG_DOF = 3
LEN_HIST = 5
MODEL_IN_SIZE = 2 * LEG_DOF * LEN_HIST

class UniNet(nn.Module):
    def __init__(self, model):
        super(UniNet, self).__init__()
        self.core_model = model

    def forward(self, x): # x: 4 * MODEL_IN_SIZE;
        # x: q_err_hip_1, dq_hip_1, q_err_thigh_1, dq_thigh_1, q_err_calf_1, dq_calf_1 (30)
        #    q_err_hip_2, dq_hip_2, q_err_thigh_2, dq_thigh_2, q_err_calf_2, dq_calf_2 (30)
        out = torch.tensor(()).to(x.device)
        for i in range(LEG_NUM):
            sub_in = x[:, MODEL_IN_SIZE*i:MODEL_IN_SIZE*(i+1)]
            sub_out = self.core_model(sub_in)
            out = torch.cat((out, sub_out), 1)
        return out

class Go1(LeggedRobot):
    cfg: Go1RoughCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            sub_model = torch.jit.load(actuator_network_path).to(self.device)
            self.actuator_network = UniNet(sub_model)

        # get mean and std of input and output from data
        self.pos_err_mean = torch.tile(torch.tensor([0.00036437, 0.01540757, -0.00972657]), (LEG_NUM, )).to(self.device)
        self.pos_err_std = torch.tile(torch.tensor([0.11722939, 0.19275887, 0.28700321]), (LEG_NUM, )).to(self.device)
        self.vel_mean = torch.tile(torch.tensor([-0.00017714, -0.00024455,  0.0005956 ]), (LEG_NUM, )).to(self.device)
        self.vel_std = torch.tile(torch.tensor([2.31517027, 3.84613839, 5.52599008]), (LEG_NUM, )).to(self.device)


    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.model_ins = torch.zeros(self.num_envs, MODEL_IN_SIZE * LEG_NUM, device=self.device, requires_grad=False) # [all envs * all legs, model-in-DOF]

        # init pos err and vel buffer(12 DOF)
        self.pos_err_buffs = np.zeros((self.num_envs, self.num_actions, LEN_HIST))
        self.vel_buffs = np.zeros((self.num_envs, self.num_actions, LEN_HIST))

    def _compute_poses(self, actions):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            dVel = self.actuator_advance(actions)

            return super()._compute_poses(actions)
        else:
            # pd controller
            return super()._compute_poses(actions)

    # TODO: actuator model buffer and forward
    def actuator_advance(self, actions):
        # scale pos_err and vel TODO: clip
        pos_err = actions - self.dof_pos
        pos_err_s = (pos_err - self.pos_err_mean) / self.pos_err_std
        vel_s = (self.dof_vel - self.vel_mean) / self.vel_std

        # TODO: note that the pos_err is of 12-dim, but real model_in is od=f 3-dim
        model_in = np.array([])
        for i in range(self.num_actions):
            # fill buffers with scaled data [t-h, ... , t-0]
            # hist can be different for each joint
            pos_err_temp = np.delete(self.pos_err_buffs[:, i, :], 0, axis=1)  # need to be numpy,
            self.pos_err_buffs[:, i, :] = np.append(pos_err_temp, pos_err_s[:, i].unsqueeze(-1).cpu().numpy(), axis=1)

            vel_temp = np.delete(self.vel_buffs[:, i, :], 0, axis=1)
            self.vel_buffs[:, i, :] = np.append(vel_temp, vel_s[:, i].unsqueeze(-1).cpu().numpy(), axis=1)

            # fill actuator model input vector
            self.model_ins[:, 2 * i * LEN_HIST:(2 * i + 1) * LEN_HIST] = torch.from_numpy(self.pos_err_buffs[:, i, :])
            self.model_ins[:, (2 * i + 1) * LEN_HIST:(2 * i + 2) * LEN_HIST] = torch.from_numpy(self.vel_buffs[:, i, :])

        with torch.inference_mode():
            # advance actuator mlp
            dVel = self.actuator_network(self.model_ins)

            # upscale mlp output(the dVel mean is counteracted)
            dVel *= self.vel_std

        return dVel

    # def compute_observations(self):
    #     super().compute_observations()
    #
    #     if (pos_num_history_stack != 0 or vel_num_history_stack != 0 or action_num_history_stack != 0):
    #         self.obs_buf = torch.cat((  self.obs_buf,
    #                                     self.dof_pos_hist[:,:(self.pos_num_hist - 1) * self.num_dof] * self.obs_scales.dof_pos,
    #                                     self.dof_vel_hist[:,:(self.vel_num_hist - 1) * self.num_dof] * self.obs_scales.dof_vel,
    #                                     self.dof_action_hist[:, :(self.action_num_hist - 1) * self.num_dof] * 1.0
    #                                  ),
    #                                 dim=-1)







