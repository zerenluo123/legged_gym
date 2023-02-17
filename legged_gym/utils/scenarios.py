import numpy as np
import torch

class TestScenario:
    def __init__(self, env):
        self.control_dt = env.dt
        self.env = env
        self.t = 0.0

class HorizontalLine(TestScenario): # back-and-forth
    def __init__(self, env, avg_vel, length):
        super().__init__(env)
        self.l = length
        self.avg_vel = avg_vel
        self.traj_time = self.l / avg_vel
        self.test_step = 2 * int(self.traj_time / self.control_dt) # 2X: back-and-forth

    def reset(self):
        self.__init__(self.env, self.avg_vel, self.l)

    def advance(self):
        cmd = np.zeros(3)

        # update traj
        if self.t <= self.traj_time:                                    # forward
            velT = self.avg_vel
        elif self.t > self.traj_time and self.t <= 2 * self.traj_time:  # backward
            velT = -self.avg_vel
        else:
            velT = 0.

        self.t += self.control_dt

        cmd[0] = velT
        cmd[1] = 0.
        cmd[2] = 0. # heading

        self.env.commands[0, :2] = torch.from_numpy(cmd[:2]).to(self.env.device).to(torch.float)
        self.env.commands[0, 3] = torch.tensor(cmd[2]) # set heading to env's commands

        return self.env.commands[0, :3]  # only for logging


