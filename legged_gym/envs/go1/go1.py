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
LEN_HIST = 50
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
        self.pos_err_mean = torch.tile(torch.tensor( [0.00013177, 0.00023659, 0.01997889]), (LEG_NUM, )).to(self.device)
        self.pos_err_std = torch.tile(torch.tensor([0.09988626, 0.15470642, 0.23311185]), (LEG_NUM, )).to(self.device)
        self.vel_mean = torch.tile(torch.tensor( [ 6.23736086e-06, -3.09437409e-05,  7.87997899e-04]), (LEG_NUM, )).to(self.device)
        self.vel_std = torch.tile(torch.tensor( [1.5616826,  2.2764919,  3.34624232]), (LEG_NUM, )).to(self.device)
        self.dVel_mean = torch.tile(torch.tensor([3.69162194e-06, -1.25283373e-06, -9.83731829e-06]), (LEG_NUM, )).to(self.device)
        self.dVel_std = torch.tile(torch.tensor([0.28239333, 0.41029372, 0.65388311]), (LEG_NUM, )).to(self.device)

        self.enforce_pos_limit = True # change this for comparison



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

            # scale pos_err and vel (actionscaled = actions * scale + defaultangle, also need to be the same for the training data?)
            pos_err = actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos
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

            # target_vels = self.dof_vel + dVel
            # target_poses = self.dof_pos + self.sim_params.dt * target_vels

            target_poses = self.dof_pos + self.sim_params.dt * self.dof_vel
            # target_vels = self.dof_vel + dVel

            # if at limit, compute new veldiff at that limit (! only handle those point out of the joint limit !)
            # only allow for velocities moving away from limit
            if self.enforce_pos_limit:
                enforce_pos_limit_ids = (torch.any(target_poses <= self.dof_pos_limits[:, 0], dim=1) |
                                         torch.any(target_poses >= self.dof_pos_limits[:, 1], dim=1)).nonzero(as_tuple=False).flatten()
                # print("enforce_pos_limit_ids", enforce_pos_limit_ids.shape)
                target_vels_l, target_poses_l, actions_l = self.dof_vel[enforce_pos_limit_ids, :], \
                                                           target_poses[enforce_pos_limit_ids, :], \
                                                           actions[enforce_pos_limit_ids, :]
                pos_err_buffs_l, vel_buffs_l = self.pos_err_buffs[enforce_pos_limit_ids, :], self.vel_buffs[enforce_pos_limit_ids, :]
                model_ins_l = self.model_ins[enforce_pos_limit_ids, :]

                # set vel to 0 if out of joint limits
                target_vels_l *= ( (target_poses_l > self.dof_pos_limits[:, 0]) & (target_poses_l < self.dof_pos_limits[:, 1]) ) # dim: [nidxs, 12]
                # clip pos to limits
                target_poses_l = torch.clip(target_poses_l, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])

                pos_err_l = actions_l * self.cfg.control.action_scale + self.default_dof_pos - target_poses_l
                pos_err_l_s = (pos_err_l - self.pos_err_mean) / self.pos_err_std
                vel_l_s = (target_vels_l - self.vel_mean) / self.vel_std

                for i in range(self.num_actions):
                    # fill buffers with scaled data [t-h, ... , t-0]
                    # hist can be different for each joint
                    pos_err_temp = pos_err_buffs_l[:, i, :][:, 1:]  # delete the first column
                    pos_err_buffs_l[:, i, :] = torch.cat((pos_err_temp, pos_err_l_s[:, i].unsqueeze(-1)), dim=1)

                    vel_temp = vel_buffs_l[:, i, :][:, 1:]
                    vel_buffs_l[:, i, :] = torch.cat((vel_temp, vel_l_s[:, i].unsqueeze(-1)), dim=1)

                    # fill actuator model input vector
                    model_ins_l[:, 2 * i * LEN_HIST:(2 * i + 1) * LEN_HIST] = pos_err_buffs_l[:, i, :]
                    model_ins_l[:, (2 * i + 1) * LEN_HIST:(2 * i + 2) * LEN_HIST] = vel_buffs_l[:, i, :]

                with torch.inference_mode():
                    # advance actuator mlp
                    dVel_l = self.actuator_network(model_ins_l)    # dim: [nidxs, 12]
                    # upscale mlp output
                    dVel_l *= self.dVel_std
                    dVel_l += self.dVel_mean

                    # only allow for vel moving away from limit
                    mask_lower = target_poses_l <= self.dof_pos_limits[:, 0] # not out-of-lower-limit: 0; out-of-lower-limit: 1
                    mask_upper = target_poses_l >= self.dof_pos_limits[:, 1] # not out-of-upper-limit: 0; out-of-upper-limit: 1
                    mask_lower_filter = torch.gt(dVel_l, torch.zeros_like(dVel_l)) # higher than 0: preserved; lower than 0: 0
                    mask_upper_filter = ~mask_lower_filter

                    dVel_lower_filter = dVel_l * mask_lower_filter * mask_lower  # higher than 0: preserved; lower than 0: 0
                    dVel_l *= ~mask_lower                                        # not out-of-lower-limit: preserved; out-of-lower-limit: 0
                    dVel_l += dVel_lower_filter

                    dVel_upper_filter = dVel_l * mask_upper_filter * mask_upper   # lower than 0: preserved; higher than 0: 0
                    dVel_l *= ~mask_upper                                         # not out-of-upper-limit: preserved; out-of-upper-limit: 0
                    dVel_l += dVel_upper_filter

                    # push the modified "out-of-limit" value back to original value
                    dVel[enforce_pos_limit_ids, :] = dVel_l

                # push the modified "out-of-limit" value back to original value
                self.dof_vel[enforce_pos_limit_ids, :], target_poses[enforce_pos_limit_ids, :] = target_vels_l, target_poses_l
                self.pos_err_buffs[enforce_pos_limit_ids, :], self.vel_buffs[enforce_pos_limit_ids, :] = pos_err_buffs_l, vel_buffs_l
                self.model_ins[enforce_pos_limit_ids, :] = model_ins_l

            target_vels = self.dof_vel + dVel


            # return torch.clip(target_poses, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1]).view(self.dof_pos.shape), \
            #        target_vels.view(self.dof_vel.shape)

            return target_poses.view(self.dof_pos.shape), target_vels.view(self.dof_vel.shape)
        else:
            return super()._actuator_advance(actions)