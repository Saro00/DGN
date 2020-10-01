import math
import torch
from functools import partial

from .scalers import scale_amplification, scale_attenuation
from .eigen_agg import get_adjacency_from_gradient_of_features

EPS = 1e-5


# each aggregator is a function taking as input X (B x N x N x Din), adj (B x N x N), self_loop and device and
# returning the aggregated value of X (B x N x Din) for each dimension

def aggregate_identity(X, adj, eigvec=None, self_loop=False, device='cpu', avg_d=None):
    # Y is corresponds to the elements of the main diagonal of X
    (_, N, N, _) = X.shape
    Y = torch.sum(torch.mul(X, torch.eye(N).reshape(1, N, N, 1)), dim=2)
    return Y


def aggregate_mean(X, adj, eigvec=None, self_loop=False, device='cpu', avg_d=None):
    # D^{-1} A * X    i.e. the mean of the neighbours

    if self_loop:  # add self connections
        (B, N, _) = adj.shape
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    D = torch.sum(adj, -1, keepdim=True)
    X_sum = torch.sum(torch.mul(X, adj.unsqueeze(-1)), dim=2)
    X_mean = torch.div(X_sum, D)
    return X_mean


def aggregate_max(X, adj, eigvec=None, min_value=-math.inf, self_loop=False, device='cpu', avg_d=None):
    (B, N, N, Din) = X.shape

    if self_loop:  # add self connections
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    adj = adj.unsqueeze(-1)  # adding extra dimension
    M = torch.where(adj > 0.0, X, torch.tensor(min_value, device=device))
    max = torch.max(M, -3)[0]
    return max


def aggregate_min(X, adj, eigvec=None, max_value=math.inf, self_loop=False, device='cpu', avg_d=None):
    (B, N, N, Din) = X.shape

    if self_loop:  # add self connections
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    adj = adj.unsqueeze(-1)  # adding extra dimension
    M = torch.where(adj > 0.0, X, torch.tensor(max_value, device=device))
    min = torch.min(M, -3)[0]
    return min


def aggregate_std(X, adj, eigvec=None, self_loop=False, device='cpu', avg_d=None):
    # sqrt(relu(D^{-1} A X^2 - (D^{-1} A X)^2) + EPS)     i.e.  the standard deviation of the features of the neighbours
    # the EPS is added for the stability of the derivative of the square root
    std = torch.sqrt(aggregate_var(X, adj, self_loop, device) + EPS)  # sqrt(mean_squares_X - mean_X^2)
    return std


def aggregate_var(X, adj, eigvec=None, self_loop=False, device='cpu', avg_d=None):
    # relu(D^{-1} A X^2 - (D^{-1} A X)^2)     i.e.  the variance of the features of the neighbours

    if self_loop:  # add self connections
        (B, N, _) = adj.shape
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    D = torch.sum(adj, -1, keepdim=True)
    X_sum_squares = torch.sum(torch.mul(torch.mul(X, X), adj.unsqueeze(-1)), dim=2)
    X_mean_squares = torch.div(X_sum_squares, D)  # D^{-1} A X^2
    X_mean = aggregate_mean(X, adj)  # D^{-1} A X
    var = torch.relu(X_mean_squares - torch.mul(X_mean, X_mean))  # relu(mean_squares_X - mean_X^2)
    return var


def aggregate_sum(X, adj, eigvec=None, self_loop=False, device='cpu', avg_d=None):
    # A * X    i.e. the mean of the neighbours

    if self_loop:  # add self connections
        (B, N, _) = adj.shape
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    X_sum = torch.sum(torch.mul(X, adj.unsqueeze(-1)), dim=2)
    return X_sum


def aggregate_normalised_mean(X, adj, eigvec=None, self_loop=False, device='cpu', avg_d=None):
    # D^{-1/2] A D^{-1/2] X
    (B, N, N, _) = X.shape

    if self_loop:  # add self connections
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    rD = torch.mul(torch.pow(torch.sum(adj, -1, keepdim=True), -0.5), torch.eye(N, device=device)
                   .unsqueeze(0).repeat(B, 1, 1))  # D^{-1/2]
    adj = torch.matmul(torch.matmul(rD, adj), rD)  # D^{-1/2] A' D^{-1/2]

    X_sum = torch.sum(torch.mul(X, adj.unsqueeze(-1)), dim=2)
    return X_sum


def aggregate_softmax(X, adj, eigvec=None, self_loop=False, device='cpu', avg_d=None):
    # for each node sum_i(x_i*exp(x_i)/sum_j(exp(x_j)) where x_i and x_j vary over the neighbourhood of the node
    (B, N, N, Din) = X.shape

    if self_loop:  # add self connections
        adj = adj + torch.eye(N, device=device).unsqueeze(0)

    X_exp = torch.exp(X)
    adj = adj.unsqueeze(-1)  # adding extra dimension
    X_exp = torch.mul(X_exp, adj)
    X_sum = torch.sum(X_exp, dim=2, keepdim=True)
    softmax = torch.sum(torch.mul(torch.div(X_exp, X_sum), X), dim=2, avg_d=None)
    return softmax


def aggregate_softmin(X, adj, eigvec=None, self_loop=False, device='cpu', avg_d=None):
    # for each node sum_i(x_i*exp(-x_i)/sum_j(exp(-x_j)) where x_i and x_j vary over the neighbourhood of the node
    return -aggregate_softmax(-X, adj, self_loop=self_loop, device=device)


def aggregate_moment_rooted(X, adj, eigvec=None, self_loop=False, device='cpu', n=3, avg_d=None):
    if self_loop:  # add self connections
        (B, N, _) = adj.shape
        adj = adj + torch.eye(N, device=device).unsqueeze(0)
    D = torch.sum(adj, -1, keepdim=True)
    X_mean = aggregate_mean(X, adj, self_loop=self_loop, device=device)
    X_n = torch.div(torch.sum(torch.mul(torch.pow(X - X_mean.unsqueeze(2), n), adj.unsqueeze(-1)), dim=2), D)
    rooted_X_n = torch.sign(X_n) * torch.pow(torch.abs(X_n) + EPS, 1. / n)
    return rooted_X_n


def aggregate_moment_div_stdn(X, adj, eigvec=None, self_loop=False, device='cpu', n=3, avg_d=None):
    if self_loop:  # add self connections
        (B, N, _) = adj.shape
        adj = adj + torch.eye(N, device=device).unsqueeze(0)
    D = torch.sum(adj, -1, keepdim=True)
    X_mean = aggregate_mean(X, adj, self_loop=self_loop, device=device)
    X_n = torch.div(torch.sum(torch.mul(torch.pow(X - X_mean.unsqueeze(2), n), adj.unsqueeze(-1)), dim=2), D)\
          / (torch.pow(aggregate_std(X, adj, self_loop=self_loop, device=device), n) + EPS)
    return X_n


def aggregate_moment_2(X, adj, eigvec=None, self_loop=False, device='cpu'):
    return aggregate_moment_rooted(X, adj, self_loop=self_loop, device=device, n=2, avg_d=None)


def aggregate_moment_3(X, adj, eigvec=None, self_loop=False, device='cpu'):
    return aggregate_moment_rooted(X, adj, self_loop=self_loop, device=device, n=3, avg_d=None)


def aggregate_moment_4(X, adj, eigvec=None, self_loop=False, device='cpu'):
    return aggregate_moment_rooted(X, adj, self_loop=self_loop, device=device, n=4, avg_d=None)


def aggregate_moment_5(X, adj, self_loop=False, device='cpu'):
    return aggregate_moment_rooted(X, adj, self_loop=self_loop, device=device, n=5, avg_d=None)


def aggregate_mean_amplified(X, adj, eigvec=None, self_loop=False, device='cpu', avg_d=None):
    X = aggregate_mean(X, adj, self_loop, device)
    X = scale_amplification(X, adj, avg_d)
    return X


def aggregate_mean_attenuated(X, adj, eigvec=None, self_loop=False, device='cpu', avg_d=None):
    X = aggregate_mean(X, adj, self_loop, device)
    X = scale_attenuation(X, adj, avg_d)
    return X


def get_eig_adjacency(adj, eig_idx, eigvec, normalization='none', add_diag=True, absolute_adj=False, normalize_L=False,
                      eig_acos=True):

    # Convert the eig_idx into a list
    try:
        eig_idx = list(eig_idx)
    except:
        eig_idx = [eig_idx]

    # Generate an adjacency matrix for each eigenvectors from `eig_idx`
    eig_adj = {}
    for ii in eig_idx:
        if ii != 0:
            this_eigvec = eigvec[..., ii]
            if eig_acos:
                this_eigvec = torch.acos(this_eigvec / torch.max(torch.abs(this_eigvec)))
            w_adj = get_adjacency_from_gradient_of_features(adj, features=this_eigvec,
                                                            normalization=normalization, add_diag=add_diag,
                                                            absolute_adj=absolute_adj)
            eig_adj[ii] = w_adj

        else:
            eig_adj[ii] = adj / (torch.sum(adj.abs(), dim=-1, keepdims=True) + EPS)

    return eig_adj


def aggregate_dirs(X, adj, eigvec, eig_idx, normalization='none', add_diag=True, agg_type='derivative', normalize_L=False,
                   eig_acos=True, self_loop=False, device='cpu', avg_d=None):

    agg_type = agg_type.lower()
    if agg_type not in ['derivative', 'smoothing', 'both']:
        raise ValueError(f'Unknown agg_type "{agg_type}"')

    # Get dictionary of adjacency matrices
    adj_dict = get_eig_adjacency(adj, eig_idx, eigvec.to(device),  normalization=normalization, add_diag=add_diag,
                                 absolute_adj=False, normalize_L=normalize_L, eig_acos=eig_acos)

    X_agg = []
    for ii, this_adj in adj_dict.items():
        # Compute derivative
        if ((agg_type == 'derivative') or (agg_type == 'both')) and (ii != 0):
            this_agg = aggregate_sum(X, this_adj, self_loop=self_loop, device=device)
            X_agg.append(this_agg)

        # Compute smoothing
        if (agg_type == 'smoothing') or (agg_type == 'both') or (ii == 0):
            this_agg = aggregate_sum(X, this_adj.abs(), self_loop=self_loop, device=device)
            X_agg.append(this_agg)

    return torch.cat(X_agg, dim=-1)


dir_aggregator = partial(aggregate_dirs, normalization='row-abs', add_diag=True,
                        normalize_L=False, eig_acos=True)
AGGREGATORS = {
    # Standard aggregators
    'mean': aggregate_mean, 
    'sum': aggregate_sum, 
    'max': aggregate_max, 
    'min': aggregate_min,
    'identity': aggregate_identity, 
    'std': aggregate_std, 
    'var': aggregate_var,
    'normalised_mean': aggregate_normalised_mean, 
    'softmax': aggregate_softmax, 
    'softmin': aggregate_softmin,
    'moment3': partial(aggregate_moment_rooted, n=3),
    'moment4': partial(aggregate_moment_rooted, n=4),
    'moment5': partial(aggregate_moment_rooted, n=5),
    'mean_amplified': aggregate_mean_amplified, 
    'mean_attenuated':aggregate_mean_attenuated,
    
    # Directional aggregators
    'dir0': partial(dir_aggregator, agg_type='smoothing', eig_idx=[0]),

    'dir1-dx': partial(dir_aggregator, agg_type='derivative', eig_idx=[1], normalization='row-abs'),
    'dir2-dx': partial(dir_aggregator, agg_type='derivative', eig_idx=[1, 2], normalization='row-abs'),
    'dir3-dx': partial(dir_aggregator, agg_type='derivative', eig_idx=[1, 2, 3], normalization='row-abs'),
    'dir4-dx': partial(dir_aggregator, agg_type='derivative', eig_idx=[1, 2, 3, 4], normalization='row-abs'),
    'dir5-dx': partial(dir_aggregator, agg_type='derivative', eig_idx=[1, 2, 3, 4, 5], normalization='row-abs'),


    'dir1-smooth': partial(dir_aggregator, agg_type='smoothing', eig_idx=[1], normalization='row-abs'),
    'dir2-smooth': partial(dir_aggregator, agg_type='smoothing', eig_idx=[1, 2], normalization='row-abs'),
    'dir3-smooth': partial(dir_aggregator, agg_type='smoothing', eig_idx=[1, 2, 3], normalization='row-abs'),
    'dir4-smooth': partial(dir_aggregator, agg_type='smoothing', eig_idx=[1, 2, 3, 4]),
    'dir5-smooth': partial(dir_aggregator, agg_type='smoothing', eig_idx=[1, 2, 3, 4, 5]),

    'dir1-both': partial(dir_aggregator, agg_type='both', eig_idx=[1]),
    'dir2-both': partial(dir_aggregator, agg_type='both', eig_idx=[1, 2]),
    'dir3-both': partial(dir_aggregator, agg_type='both', eig_idx=[1, 2, 3]),
    'dir4-both': partial(dir_aggregator, agg_type='both', eig_idx=[1, 2, 3, 4]),
    'dir5-both': partial(dir_aggregator, agg_type='both', eig_idx=[1, 2, 3, 4, 5])

    }
