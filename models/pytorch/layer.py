import torch
import torch.nn as nn

from .aggregators import AGGREGATORS
from .scalers import SCALERS
from .layers import FCLayer, MLP


class EIGTower(nn.Module):
    def __init__(self, in_features, out_features, aggregators, scalers, avg_d, self_loop, eigs, pretrans_layers,
                 posttrans_layers, device):
        """
        :param in_features:     size of the input per node of the tower
        :param out_features:    size of the output per node of the tower
        :param aggregators:     set of aggregation functions each taking as input X (B x N x N x Din), adj (B x N x N), self_loop and device
        :param scalers:         set of scaling functions each taking as input X (B x N x Din), adj (B x N x N) and avg_d
        """
        super(EIGTower, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.aggregators = aggregators
        self.scalers = scalers
        self.self_loop = self_loop
        self.eigs = eigs

        agg_test = torch.cat([agg(torch.ones(1, 10, 10, 1, device=device), torch.rand(1, 10, 10, device=device),
                                  eigvec=torch.rand(1, 10, 23, device=device), device=self.device) for agg in self.aggregators], dim=-1)

        self.pretrans = MLP(in_size=2 * self.in_features, hidden_size=self.in_features, out_size=self.in_features,
                            layers=pretrans_layers, mid_activation='relu', last_activation='none')

        self.posttrans = MLP(in_size=((agg_test.shape[-1]) * len(scalers) + 1) * self.in_features, hidden_size=self.out_features,
                                 out_size=self.out_features, layers=posttrans_layers, mid_activation='relu', last_activation='none')

        self.avg_d = avg_d

    def forward(self, input, adj, eigvec):
        (B, N, _) = adj.shape

        # pre-aggregation transformation
        h_i = input.unsqueeze(2).repeat(1, 1, N, 1)
        h_j = input.unsqueeze(1).repeat(1, N, 1, 1)
        h_cat = torch.cat([h_i, h_j], dim=3)
        h_mod = self.pretrans(h_cat)
        adj = adj.to(self.device)
        # aggregation
        m = torch.cat([aggregate(h_mod, adj, eigvec=eigvec, self_loop=self.self_loop, device=self.device, avg_d=self.avg_d) for aggregate in self.aggregators],dim=2)
        m = torch.cat([scale(m, adj, avg_d=self.avg_d) for scale in self.scalers], dim=2)
        m_cat = torch.cat([input, m], dim=2)
        out = self.posttrans(m_cat)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class EIGLayer(nn.Module):

    def __init__(self, in_features, out_features, aggregators, scalers, NN_eig, avg_d, eigs, towers=1, self_loop=False,
                 pretrans_layers=1, posttrans_layers=1, divide_input=True, device='cpu'):
        """
        :param in_features:     size of the input per node
        :param out_features:    size of the output per node
        :param aggregators:     set of aggregation function identifiers
        :param scalers:         set of scaling functions identifiers
        :param avg_d:           average degree of nodes in the training set, used by scalers to normalize
        :param self_loop:       whether to add a self loop in the adjacency matrix when aggregating
        :param pretrans_layers: number of layers in the transformation before the aggregation
        :param posttrans_layers: number of layers in the transformation after the aggregation
        :param divide_input:    whether the input features should be split between towers or not
        :param device:          device used for computation
        """
        super(EIGLayer, self).__init__()
        assert ((not divide_input) or in_features % towers == 0), "if divide_input is set the number of towers has to divide in_features"
        assert (out_features % towers == 0), "the number of towers has to divide the out_features"

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        scalers = [SCALERS[scale] for scale in scalers]

        self.divide_input = divide_input
        self.input_tower = in_features // towers if divide_input else in_features
        self.output_tower = out_features // towers

        # convolution
        self.towers = nn.ModuleList()
        for _ in range(towers):
            self.towers.append(EIGTower(in_features=self.input_tower, out_features=self.output_tower, aggregators=aggregators,
                                        scalers=scalers, avg_d=avg_d, self_loop=self_loop, eigs=eigs, pretrans_layers=pretrans_layers,
                                        posttrans_layers=posttrans_layers, device=device))
        # mixing network
        self.mixing_network = FCLayer(out_features, out_features, activation='LeakyReLU')

    def forward(self, input, adj, eigvec=None):
        # convolution
        if self.divide_input:
            y = torch.cat(
                [tower(input[:, :, n_tower * self.input_tower: (n_tower + 1) * self.input_tower], adj, eigvec)
                 for n_tower, tower in enumerate(self.towers)], dim=2)
        else:
            y = torch.cat([tower(input, adj, eigvec) for tower in self.towers], dim=2)

        # mixing network
        return self.mixing_network(y)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'