EPS = 1e-5
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregators import AGGREGATORS
from .layers import MLP, FCLayer
from .scalers import SCALERS
from dgl.nn.pytorch.glob import mean_nodes, sum_nodes


class VirtualNode(nn.Module):
    def __init__(self, dim, dropout, batch_norm=False, bias=True, residual=True, vn_type='mean'):
        super().__init__()
        self.vn_type = vn_type.lower()
        self.fc_layer = FCLayer(in_size=dim, out_size=dim, activation='relu', dropout=dropout,
                                b_norm=batch_norm, bias=bias)
        self.residual = residual

    def forward(self, g, h, vn_h):

        g.ndata['h'] = h

        # Pool the features
        if self.vn_type == 'mean':
            pool = mean_nodes(g, 'h')
        elif self.vn_type == 'sum':
            pool = sum_nodes(g, 'h')
        elif self.vn_type == 'logsum':
            pool = mean_nodes(g, 'h')
            lognum = torch.log(torch.tensor(g.batch_num_nodes, dtype=h.dtype, device=h.device))
            pool = pool * lognum.unsqueeze(-1)
        else:
            raise ValueError(f'Undefined input "{self.pooling}". Accepted values are "sum", "mean", "logsum"')

        # Compute the new virtual node features
        vn_h_temp = self.fc_layer.forward(vn_h + pool)
        if self.residual:
            vn_h = vn_h + vn_h_temp
        else:
            vn_h = vn_h_temp

        # Add the virtual node value to the graph features
        temp_h = torch.cat(
            [vn_h[ii:ii + 1].repeat(num_nodes, 1) for ii, num_nodes in enumerate(g.batch_num_nodes)],
            dim=0)
        h = h + temp_h

        return vn_h, h


class DGNLayerComplex(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d, residual,
                 edge_features, edge_dim, pretrans_layers=1, posttrans_layers=1):
        super().__init__()

        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_features = edge_features
        self.residual = residual

        self.aggregators = aggregators
        self.scalers = scalers

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.pretrans = MLP(in_size=2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim,
                            out_size=in_dim, layers=pretrans_layers, mid_activation='relu', last_activation='none')
        self.posttrans = MLP(in_size=(len(aggregators) * len(scalers) + 1) * in_dim, hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d
        if in_dim != out_dim:
            self.residual = False

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2 = torch.cat([edges.src['h'], edges.dst['h'], edges.data['ef']], dim=1)
        else:
            z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        return {'e': self.pretrans(z2), 'eig_s': edges.src['eig'], 'eig_d': edges.dst['eig']}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'eig_s': edges.data['eig_s'].to('cuda' if torch.cuda.is_available() else 'cpu'),
                'eig_d': edges.data['eig_d'].to('cuda' if torch.cuda.is_available() else 'cpu')}

    def reduce_func(self, nodes):
        h_in = nodes.data['h']
        h = nodes.mailbox['e']
        eig_s = nodes.mailbox['eig_s']
        eig_d = nodes.mailbox['eig_d']
        D = h.shape[-2]

        # aggregators and scalers
        h = torch.cat([aggregate(h, eig_s, eig_d, h_in) for aggregate in self.aggregators], dim=1)
        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)

        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):

        h_in = h
        g.ndata['h'] = h

        if self.edge_features:  # add the edges information only if edge_features = True
            g.edata['ef'] = e

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization and residual
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h

        h = F.dropout(h, self.dropout, training=self.training)

        return h


class DGNLayerSimple(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, residual, avg_d,
                 posttrans_layers=1):
        super().__init__()

        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.aggregators = aggregators
        self.scalers = scalers
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.posttrans = MLP(in_size=(len(aggregators) * len(scalers)) * in_dim, hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d
        if in_dim != out_dim:
            self.residual = False

    def pretrans_edges(self, edges):
        return {'e': edges.src['h'], 'eig_s': edges.src['eig'], 'eig_d': edges.dst['eig']}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'eig_s': edges.data['eig_s'].to('cuda' if torch.cuda.is_available() else 'cpu'),
                'eig_d': edges.data['eig_d'].to('cuda' if torch.cuda.is_available() else 'cpu')}

    def reduce_func(self, nodes):
        h_in = nodes.data['h']
        h = nodes.mailbox['e']
        eig_s = nodes.mailbox['eig_s']
        eig_d = nodes.mailbox['eig_d']
        D = h.shape[-2]

        # aggregators and scalers
        h = torch.cat([aggregate(h, eig_s, eig_d, h_in) for aggregate in self.aggregators], dim=1)
        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)

        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):
        h_in = h
        g.ndata['h'] = h

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization and residual
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h

        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DGNTower(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d,
                 pretrans_layers, posttrans_layers, edge_features, edge_dim):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_features = edge_features

        self.aggregators = aggregators
        self.scalers = scalers

        self.batchnorm_h = nn.BatchNorm1d(out_dim)

        self.pretrans = MLP(in_size=2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim,
                            out_size=in_dim, layers=pretrans_layers, mid_activation='relu', last_activation='none')

        self.posttrans = MLP(in_size=(len(aggregators) * len(scalers) + 1) * in_dim, hidden_size=out_dim,
                             out_size=out_dim, layers=posttrans_layers, mid_activation='relu', last_activation='none')
        self.avg_d = avg_d

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2 = torch.cat([edges.src['h'], edges.dst['h'], edges.data['ef']], dim=1)
        else:
            z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        return {'e': self.pretrans(z2), 'eig_s': edges.src['eig'], 'eig_d': edges.dst['eig']}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'eig_s': edges.data['eig_s'].to('cuda' if torch.cuda.is_available() else 'cpu'),
                'eig_d': edges.data['eig_d'].to('cuda' if torch.cuda.is_available() else 'cpu')}

    def reduce_func(self, nodes):
        h_in = nodes.data['h']
        h = nodes.mailbox['e']
        eig_s = nodes.mailbox['eig_s']
        eig_d = nodes.mailbox['eig_d']
        D = h.shape[-2]

        # aggregators and scalers
        h = torch.cat([aggregate(h, eig_s, eig_d, h_in) for aggregate in self.aggregators], dim=1)
        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)

        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):
        g.ndata['h'] = h

        if self.edge_features:  # add the edges information only if edge_features = True
            g.edata['ef'] = e

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class DGNLayerTower(nn.Module):
    def __init__(self, in_dim, out_dim, aggregators, scalers, avg_d, dropout, graph_norm, batch_norm, towers=5,
                 pretrans_layers=1, posttrans_layers=1, divide_input=True, residual=False, edge_features=False,
                 edge_dim=0):
        super().__init__()
        assert ((not divide_input) or in_dim % towers == 0), "if divide_input is set the number of towers has to divide in_dim"
        assert (out_dim % towers == 0), "the number of towers has to divide the out_dim"
        assert avg_d is not None

        self.divide_input = divide_input
        self.input_tower = in_dim // towers if divide_input else in_dim
        self.output_tower = out_dim // towers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.residual = residual

        if in_dim != out_dim:
            self.residual = False

        # convolution
        self.towers = nn.ModuleList()
        for _ in range(towers):
            self.towers.append(DGNTower(in_dim=self.input_tower, out_dim=self.output_tower, aggregators=aggregators,
                                        scalers=scalers, avg_d=avg_d, pretrans_layers=pretrans_layers,
                                        posttrans_layers=posttrans_layers, batch_norm=batch_norm, dropout=dropout,
                                        graph_norm=graph_norm, edge_features=edge_features, edge_dim=edge_dim))
        # mixing network
        self.mixing_network = FCLayer(out_dim, out_dim, activation='LeakyReLU')

    def forward(self, g, h, e, snorm_n):
        h_in = h  # for residual connection

        if self.divide_input:
            h_cat = torch.cat([tower(g, h[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower], e, snorm_n)
                               for n_tower, tower in enumerate(self.towers)], dim=1)
        else:
            h_cat = torch.cat([tower(g, h, e, snorm_n) for tower in self.towers], dim=1)

        if len(self.towers) > 1:
            h_out = self.mixing_network(h_cat)
        else:
            h_out = h_cat

        if self.residual:
            h_out = h_in + h_out  # residual connection
        return h_out


class DGNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d, type_net,
                 residual, towers=5, divide_input=True, edge_features=None, edge_dim=None, pretrans_layers=1,
                 posttrans_layers=1):
        super().__init__()

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in aggregators.split()]
        scalers = [SCALERS[scale] for scale in scalers.split()]

        if type_net == 'simple':
            self.model = DGNLayerSimple(in_dim=in_dim, out_dim=out_dim, dropout=dropout, graph_norm=graph_norm,
                                        batch_norm=batch_norm, residual=residual, aggregators=aggregators,
                                        scalers=scalers, avg_d=avg_d, posttrans_layers=posttrans_layers)
        elif type_net == 'complex':
            self.model = DGNLayerComplex(in_dim=in_dim, out_dim=out_dim, dropout=dropout, graph_norm=graph_norm,
                                         batch_norm=batch_norm, aggregators=aggregators, residual=residual,
                                         scalers=scalers, avg_d=avg_d, edge_features=edge_features, edge_dim=edge_dim,
                                         pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers)
        elif type_net == 'towers':
            self.model = DGNLayerTower(in_dim=in_dim, out_dim=out_dim, aggregators=aggregators, scalers=scalers,
                                       avg_d=avg_d, dropout=dropout, graph_norm=graph_norm,
                                       batch_norm=batch_norm, towers=towers, pretrans_layers=pretrans_layers,
                                       posttrans_layers=posttrans_layers, divide_input=divide_input,
                                       residual=residual, edge_features=edge_features, edge_dim=edge_dim)
