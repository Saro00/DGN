import torch
from torch import nn
from functools import partial

EPS = 1e-5


def aggregate_mean(self, h, eig_s, eig_d):
    return torch.mean(h, dim=1)


def aggregate_max(self, h, eig_s, eig_d):
    return torch.max(h, dim=1)[0]


def aggregate_min(self, h, eig_s, eig_d):
    return torch.min(h, dim=1)[0]


def aggregate_std(self, h, eig_s, eig_d):
    return torch.sqrt(aggregate_var(self, h, eig_s, eig_d) + EPS)


def aggregate_var(self, h, eig_s, eig_d):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_sum(self, h, eig_s, eig_d):
    return torch.sum(h, dim=1)


def aggregate_eig_smooth(self, h, eig_s, eig_d, eig_idx):
    h_mod = torch.mul(h, (torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) /
                          (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), keepdim=True,
                                     dim=1))).unsqueeze(-1))
    return torch.sum(h_mod, dim=1)


def aggregate_eig_softmax(self, h, eig_s, eig_d, eig_idx, alpha):
    h_mod = torch.mul(h, torch.nn.Softmax(1)(
        alpha * (torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx])).unsqueeze(-1)))
    return torch.sum(h_mod, dim=1)


def aggregate_eig_dx(self, h, eig_s, eig_d, h_in, eig_idx):
    eig_w = ((eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) /
             (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), keepdim=True, dim=1))).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)


def aggregate_eig_dx_no_abs(self, h, eig_s, eig_d, h_in, eig_idx):
    eig_w = ((eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]) /
             (torch.sum(torch.abs(eig_s[:, :, eig_idx] - eig_d[:, :, eig_idx]), keepdim=True, dim=1) + EPS)).unsqueeze(
        -1)
    h_mod = torch.mul(h, eig_w)
    return torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in


def aggregate_eig_dx_balanced(self, h, eig_s, eig_d, h_in, eig_idx):
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
               'eig1-smooth': partial(aggregate_eig_smooth, eig_idx=1),
               'eig2-smooth': partial(aggregate_eig_smooth, eig_idx=2),
               'eig3-smooth': partial(aggregate_eig_smooth, eig_idx=3),
               'eig1-0.1': partial(aggregate_eig_softmax, eig_idx=1, alpha=0.1),
               'eig2-0.1': partial(aggregate_eig_softmax, eig_idx=2, alpha=0.1),
               'eig3-0.1': partial(aggregate_eig_softmax, eig_idx=3, alpha=0.1),
               'eig1-neg-0.1': partial(aggregate_eig_softmax, eig_idx=1, alpha=-0.1),
               'eig2-neg-0.1': partial(aggregate_eig_softmax, eig_idx=2, alpha=-0.1),
               'eig3-neg-0.1': partial(aggregate_eig_softmax, eig_idx=3, alpha=-0.1),
               'eig1-dx': partial(aggregate_eig_dx, eig_idx=1), 'eig2-dx': partial(aggregate_eig_dx, eig_idx=2),
               'eig3-dx': partial(aggregate_eig_dx, eig_idx=3),
               'eig1-dx-no-abs': partial(aggregate_eig_dx_no_abs, eig_idx=1),
               'eig2-dx-no-abs': partial(aggregate_eig_dx_no_abs, eig_idx=2),
               'eig3-dx-no-abs': partial(aggregate_eig_dx_no_abs, eig_idx=3),
               'eig1-dx-balanced': partial(aggregate_eig_dx_balanced, eig_idx=1),
               'eig2-dx-balanced': partial(aggregate_eig_dx_balanced, eig_idx=2),
               'eig3-dx-balanced': partial(aggregate_eig_dx_balanced, eig_idx=3)}
