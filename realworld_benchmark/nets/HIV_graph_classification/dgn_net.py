import torch.nn as nn
import dgl
from nets.dgn_layer import DGNLayer
from nets.mlp_readout_layer import MLPReadout
import torch
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder






class DGNNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
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
        device = net_params['device']
        self.device = device

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)

        if self.edge_feat:
            self.embedding_e = BondEncoder(emb_dim=edge_dim)

        self.layers = nn.ModuleList([DGNLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                      batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                      scalers=self.scalers, avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat,
                      edge_dim=edge_dim, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model for _
             in range(n_layers - 1)])
        self.layers.append(DGNLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    graph_norm=self.graph_norm, batch_norm=self.batch_norm,
                                    residual=self.residual, aggregators=self.aggregators, scalers=self.scalers,
                                    avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat,
                                    edge_dim=edge_dim,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model)

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
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes

        return self.MLP_layer(hg)

    def loss(self, scores, labels):
        loss = torch.nn.BCEWithLogitsLoss()(scores, labels.type(torch.FloatTensor).to('cuda').unsqueeze(-1))
        return loss