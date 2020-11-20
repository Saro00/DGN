import torch.nn as nn
import torch
import dgl
from nets.dgn_layer import DGNLayer
from nets.mlp_readout_layer import MLPReadout


class DGNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.type_net = net_params['type_net']
        self.pos_enc_dim = net_params['pos_enc_dim']
        if self.pos_enc_dim > 0:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, hidden_dim)
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']
        self.device = net_params['device']

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)
        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, edge_dim)

        self.layers = nn.ModuleList(
            [DGNLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                      batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                      scalers=self.scalers, avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat,
                      edge_dim=edge_dim, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model for _
             in range(n_layers - 1)])
        self.layers.append(DGNLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat, edge_dim=edge_dim,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model)

        if self.readout == "directional" or self.readout == "directional_abs":
            self.MLP_layer = MLPReadout(2 * out_dim, 1)
        else:
            self.MLP_layer = MLPReadout(out_dim, 1)  # 1 out dim since regression problem

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        if self.pos_enc_dim > 0:
            h_pos_enc = self.embedding_pos_enc(g.ndata['pos_enc'].to(self.device))
            h = h + h_pos_enc
        if self.edge_feat:
            e = self.embedding_e(e)

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            h = h_t
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        elif self.readout == "directional_abs":
            g.ndata['dir'] = h * torch.abs(g.ndata['eig'][:, 1:2].to(self.device)) / torch.sum(
                torch.abs(g.ndata['eig'][:, 1:2].to(self.device)), dim=1, keepdim=True)
            hg = torch.cat([dgl.mean_nodes(g, 'dir'), dgl.mean_nodes(g, 'h')], dim=1)
        elif self.readout == "directional":
            g.ndata['dir'] = h * g.ndata['eig'][:, 1:2].to(self.device) / torch.sum(
                torch.abs(g.ndata['eig'][:, 1:2].to(self.device)), dim=1, keepdim=True)
            hg = torch.cat([torch.abs(dgl.mean_nodes(g, 'dir')), dgl.mean_nodes(g, 'h')], dim=1)
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss
