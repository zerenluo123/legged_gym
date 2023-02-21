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
from .arclabdog_config import ArclabdogRoughCfg

LEG_NUM = 4
LEG_DOF = 3
LEN_HIST = 5
MODEL_IN_SIZE = 2 * LEG_DOF * LEN_HIST

class Arclabdog(LeggedRobot):
    cfg: ArclabdogRoughCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)


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
        # pd controller
        return super()._compute_poses(actions)





