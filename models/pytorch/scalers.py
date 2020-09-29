import torch

# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output

def scale_identity(X, adj, avg_d=None):
    return X


def scale_ampl_log1(X, adj, avg_d=None):
    # log(d^{-1} D + 1) * X     where d is the average degree in the training set
    D = torch.sum(adj, -1)
    scale = torch.log(torch.mul(1 / avg_d["lin"], D) + 1).unsqueeze(-1)
    X_scaled = torch.mul(scale, X)
    return X_scaled


def scale_att_exp1(X, adj, avg_d=None):
    # exp(d D^{-1} - d) * X     where d is the average degree in the training set
    D = torch.sum(adj, -1)
    scale = torch.exp(torch.div(avg_d["lin"], D) - avg_d["lin"]).unsqueeze(-1)
    X_scaled = torch.mul(scale, X)
    return X_scaled


def scale_amplification(X, adj, avg_d=None):
    # log(D + 1) / d * X     where d is the average of the ``log(D + 1)`` in the training set
    D = torch.sum(adj, -1)
    scale = (torch.log(D + 1) / avg_d["log"]).unsqueeze(-1)
    X_scaled = torch.mul(scale, X)
    return X_scaled


def scale_att_exp2(X, adj, avg_d=None):
    # (exp(D^{-1}) - 1) / d * X     where d is the average of the ``exp(D^-1)-1`` in the training set
    D = torch.sum(adj, -1)
    scale = ((torch.exp(torch.div(1, D)) - 1) / avg_d["exp"]).unsqueeze(-1)
    X_scaled = torch.mul(scale, X)
    return X_scaled


def scale_attenuation(X, adj, avg_d=None):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    D = torch.sum(adj, -1)
    scale = (avg_d["log"] / torch.log(D + 1)).unsqueeze(-1)
    X_scaled = torch.mul(scale, X)
    return X_scaled


def scale_linear(X, adj, avg_d=None):
    # d^{-1} D X     where d is the average degree in the training set
    D = torch.sum(adj, -1, keepdim=True)
    X_scaled = D * X / avg_d["lin"]
    return X_scaled


def scale_inverse_linear(X, adj, avg_d=None):
    # d D^{-1} X     where d is the average degree in the training set
    D = torch.sum(adj, -1, keepdim=True)
    X_scaled = avg_d["lin"] * X / D
    return X_scaled


SCALERS = {'identity': scale_identity, 'log': scale_ampl_log1, 'exp':scale_att_exp1,
            'log2': scale_amplification, 'exp2':scale_att_exp2, 'exp3':scale_attenuation,
            'linear': scale_linear, 'inverse_linear': scale_inverse_linear,
            'amplification': scale_amplification, 'attenuation': scale_attenuation}

    