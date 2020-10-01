import torch
from torch import nn
from functools import partial

EPS = 1e-8


def aggregate_mean(h, eig_s, eig_d, h_in):
    return torch.mean(h, dim=1)


def aggregate_max(h, eig_s, eig_d, h_in):
    return torch.max(h, dim=1)[0]


def aggregate_min(h, eig_s, eig_d, h_in):
    return torch.min(h, dim=1)[0]


def aggregate_std(h, eig_s, eig_d, h_in):
    return torch.sqrt(aggregate_var(h, eig_s, eig_d, h_in) + EPS)


def aggregate_var(h, eig_s, eig_d, h_in):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_sum(h, eig_s, eig_d, h_in):
    return torch.sum(h, dim=1)


def aggregate_dir_smooth(h, eig_s, eig_d, h_in, eig_idx):
    h_mod = torch.mul(h, (torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) /
                          (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), keepdim=True,
                                     dim=1) + EPS)).unsqueeze(-1))
    return torch.sum(h_mod, dim=1)


def aggregate_dir_softmax(h, eig_s, eig_d, h_in, eig_idx, alpha):
    h_mod = torch.mul(h, torch.nn.Softmax(1)(
        alpha * (torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx])).unsqueeze(-1)))
    return torch.sum(h_mod, dim=1)


def aggregate_dir_dx(h, eig_s, eig_d, h_in, eig_idx):
    eig_w = ((eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) /
             (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)


def aggregate_dir_dx_no_abs(h, eig_s, eig_d, h_in, eig_idx):
    eig_w = ((eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) /
             (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), keepdim=True, dim=1) + EPS)).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in


def aggregate_dir_dx_balanced(h, eig_s, eig_d, h_in, eig_idx):
    eig_front = (torch.relu(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) /
                 (torch.sum(torch.abs(torch.relu(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx])), keepdim=True,
                            dim=1) + EPS)).unsqueeze(-1)
    eig_back = (torch.relu(eig_d[:, :, eig_idx] - eig_s[:, :, eig_idx]) /
                (torch.sum(torch.abs(-torch.relu(eig_d[:, :, eig_idx] - eig_s[:, :, eig_idx])), keepdim=True,
                           dim=1) + EPS)).unsqueeze(-1)
    eig_w = (eig_front + eig_back) / 2
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)


AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std, 'var': aggregate_var,
               'dir1-smooth': partial(aggregate_dir_smooth, eig_idx=1),
               'dir2-smooth': partial(aggregate_dir_smooth, eig_idx=2),
               'dir3-smooth': partial(aggregate_dir_smooth, eig_idx=3),
               'dir1-0.1': partial(aggregate_dir_softmax, eig_idx=1, alpha=0.1),
               'dir2-0.1': partial(aggregate_dir_softmax, eig_idx=2, alpha=0.1),
               'dir3-0.1': partial(aggregate_dir_softmax, eig_idx=3, alpha=0.1),
               'dir1-neg-0.1': partial(aggregate_dir_softmax, eig_idx=1, alpha=-0.1),
               'dir2-neg-0.1': partial(aggregate_dir_softmax, eig_idx=2, alpha=-0.1),
               'dir3-neg-0.1': partial(aggregate_dir_softmax, eig_idx=3, alpha=-0.1),
               'dir1-dx': partial(aggregate_dir_dx, eig_idx=1), 
               'dir2-dx': partial(aggregate_dir_dx, eig_idx=2),
               'dir3-dx': partial(aggregate_dir_dx, eig_idx=3),
               'dir1-dx-no-abs': partial(aggregate_dir_dx_no_abs, eig_idx=1),
               'dir2-dx-no-abs': partial(aggregate_dir_dx_no_abs, eig_idx=2),
               'dir3-dx-no-abs': partial(aggregate_dir_dx_no_abs, eig_idx=3),
               'dir1-dx-balanced': partial(aggregate_dir_dx_balanced, eig_idx=1),
               'dir2-dx-balanced': partial(aggregate_dir_dx_balanced, eig_idx=2),
               'dir3-dx-balanced': partial(aggregate_dir_dx_balanced, eig_idx=3)}
