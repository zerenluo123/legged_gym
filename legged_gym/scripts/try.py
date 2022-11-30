import torch


dof_pos = torch.tensor([[1., 1., 1.7, -1., 1., 1., 0.15, 1., 1., 1.9, 1., 1.],
                        [2., 2., 2., 2., -2., 2., 2., 0.2, 4., 2., 2., 2.],
                        [0.17, 1., 1., 0.4, 1., 1., 1., 1., -1., 1., 1., 1.]])

pos_limit_low = torch.tensor([0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
pos_limit_high = torch.tensor([2.5, 2.5, 1.8, 1.9, 2.5, 2.5, 2.5, 2.5, 2.5, 1.6, 2.5, 2.5])

reset_buf = torch.any(dof_pos >= pos_limit_high, dim=1)
reset_buf2 = torch.any(dof_pos <= pos_limit_low, dim=1)

reset_buf |= reset_buf2
print(reset_buf)

print((dof_pos < pos_limit_high))
print((dof_pos > pos_limit_low))
print((dof_pos < pos_limit_high) & (dof_pos > pos_limit_low))
a = 8 * torch.ones(3, 12)
print(a * ((dof_pos < pos_limit_high) & (dof_pos > pos_limit_low)))

reset_all = torch.any(torch.any(dof_pos >= pos_limit_high, dim=1)) or \
            torch.any(torch.any(dof_pos <= pos_limit_low, dim=1))
if (torch.any(torch.any(dof_pos >= pos_limit_high, dim=1)) or torch.any(torch.any(dof_pos <= pos_limit_low, dim=1))):
    print("there are joint exceeding limit!!")
# time_out_buf = pos_limit_low > 1  # no terminal reward for time-outs
# print(time_out_buf)

mask_lower = dof_pos <= pos_limit_low # not out-of-lower-limit: 0; out-of-lower-limit: 1
mask_upper = dof_pos >= pos_limit_high
mask_lower_filter = torch.gt(dof_pos, torch.zeros_like(dof_pos))
mask_upper_filter = torch.gt(torch.zeros_like(dof_pos), dof_pos)
print(mask_lower_filter)
print(mask_lower)
# print(mask_upper_filter)
# print(~mask_lower_filter)
print(dof_pos * mask_lower_filter * mask_lower)

