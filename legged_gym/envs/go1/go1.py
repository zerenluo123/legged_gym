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
        self.pos_err_mean = torch.tile(torch.tensor([0.00014115, 0.00058488, 0.01920658]), (LEG_NUM, )).to(self.device)
        self.pos_err_std = torch.tile(torch.tensor([0.09986748, 0.15384995, 0.23037557]), (LEG_NUM, )).to(self.device)
        self.vel_mean = torch.tile(torch.tensor([-0.00013513, -0.00010417,  0.00087039]), (LEG_NUM, )).to(self.device)
        self.vel_std = torch.tile(torch.tensor([1.56212732, 2.27749993, 3.34788431]), (LEG_NUM, )).to(self.device)
        self.dVel_mean = torch.tile(torch.tensor([3.69162194e-06, -1.25283373e-06, -9.83731829e-06]), (LEG_NUM, )).to(self.device)
        self.dVel_std = torch.tile(torch.tensor([0.28239333, 0.41029372, 0.65388311]), (LEG_NUM, )).to(self.device)



    def reset_idx(self, env_ids): # the only way to get pos and vel from env. other state rolls forward internally
        super().reset_idx(env_ids)

        self.act_pos[env_ids] = self.dof_pos[env_ids]
        self.act_vel[env_ids] = self.dof_vel[env_ids]

        # Additionaly fill actuator network history buffer states
        pos_err = - self.act_pos[env_ids]                               # self.act_pos[env_ids] ~ (12, )
        pos_err_s = (pos_err - self.pos_err_mean) / self.pos_err_std
        vel_s = (self.act_vel[env_ids] - self.vel_mean) / self.vel_std  # self.act_vel[env_ids] ~ (12, )

        for i in range(LEN_HIST):
            self.pos_err_buffs[env_ids, :, i] = pos_err_s
            self.vel_buffs[env_ids, :, i] = vel_s

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.model_ins = torch.zeros(self.num_envs, MODEL_IN_SIZE * LEG_NUM, device=self.device, requires_grad=False) # [all envs * all legs, model-in-DOF]

        # init pos err and vel buffer(12 DOF)
        # self.pos_err_buffs = np.zeros((self.num_envs, self.num_actions, LEN_HIST))
        # self.vel_buffs = np.zeros((self.num_envs, self.num_actions, LEN_HIST))
        self.pos_err_buffs = torch.zeros(self.num_envs, self.num_actions, LEN_HIST, device=self.device, requires_grad=False)
        self.vel_buffs = torch.zeros(self.num_envs, self.num_actions, LEN_HIST, device=self.device, requires_grad=False)

        self.act_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)
        self.act_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False)

    def _compute_poses(self, actions):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            # self.act_pos, _ = self._actuator_advance(actions)
            # return target_poses

            return super()._compute_poses(actions)
        else:
            # pd controller
            return super()._compute_poses(actions)

    # # TODO: actuator model buffer and forward
    # def _actuator_advance(self, actions):
    #
    #     if self.cfg.control.use_actuator_network:
    #         vel = 0.2*actions
    #         pos = self.dof_pos + self.sim_params.dt * vel
    #
    #         return torch.clip(pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1]).view(self.dof_pos.shape), \
    #                vel.view(self.dof_vel.shape)
    #
    #     else:
    #         return super()._actuator_advance(actions)

    # TODO: actuator model buffer and forward
    def _actuator_advance(self, actions):

        if self.cfg.control.use_actuator_network:

            # scale pos_err and vel TODO: clip
            pos_err = actions - self.dof_pos
            pos_err_s = (pos_err - self.pos_err_mean) / self.pos_err_std
            vel_s = (self.dof_vel - self.vel_mean) / self.vel_std

            # TODO: note that the pos_err is of 12-dim, but real model_in is od=f 3-dim
            model_in = np.array([])
            for i in range(self.num_actions):
                # fill buffers with scaled data [t-h, ... , t-0]
                # hist can be different for each joint
                pos_err_temp = self.pos_err_buffs[:, i, :][:, 1:]  # delete the first column
                self.pos_err_buffs[:, i, :] = torch.cat( (pos_err_temp, pos_err_s[:, i].unsqueeze(-1)), dim=1 )

                vel_temp = self.vel_buffs[:, i, :][:, 1:]
                self.vel_buffs[:, i, :] = torch.cat( (vel_temp, vel_s[:, i].unsqueeze(-1)), dim=1 )

                # fill actuator model input vector
                self.model_ins[:, 2 * i * LEN_HIST:(2 * i + 1) * LEN_HIST] = self.pos_err_buffs[:, i, :]
                self.model_ins[:, (2 * i + 1) * LEN_HIST:(2 * i + 2) * LEN_HIST] = self.vel_buffs[:, i, :]

            with torch.inference_mode():
                # advance actuator mlp
                dVel = self.actuator_network(self.model_ins)

                # upscale mlp output
                dVel *= self.dVel_std
                dVel += self.dVel_mean

            target_vels = self.dof_vel + dVel
            target_poses = self.dof_pos + self.sim_params.dt * target_vels

            # print("**********************")
            # print(target_poses)

            return torch.clip(target_poses, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1]).view(self.dof_pos.shape), \
                   target_vels.view(self.dof_vel.shape)

            # return super()._actuator_advance(actions)

        else:
            return super()._actuator_advance(actions)