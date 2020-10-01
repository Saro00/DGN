import torch.nn as nn
import torch
import dgl
from nets.dgn_layer import DGNLayer
from nets.mlp_readout_layer import MLPReadout


class EIGNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim_node = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.type_net = net_params['type_net']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.device = net_params['device']
        self.edge_feat = net_params['edge_feat']
        edge_dim = net_params['edge_dim']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.n_classes = n_classes
        self.aggregators = net_params['aggregators']
        self.scalers = net_params['scalers']
        self.avg_d = net_params['avg_d']
        self.residual = net_params['residual']
        pretrans_layers = net_params['pretrans_layers']
        posttrans_layers = net_params['posttrans_layers']

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([DGNLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, graph_norm=self.graph_norm,
                      batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                      scalers=self.scalers, avg_d=self.avg_d, type_net=self.type_net, edge_features=self.edge_feat,
                      edge_dim=edge_dim, pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model for _
             in range(n_layers - 1)])
        self.layers.append(DGNLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout, graph_norm=self.graph_norm,
                                    batch_norm=self.batch_norm, residual=self.residual, aggregators=self.aggregators,
                                    scalers=self.scalers, avg_d=self.avg_d, type_net=self.type_net,
                                    edge_features=self.edge_feat, edge_dim=edge_dim,
                                    pretrans_layers=pretrans_layers, posttrans_layers=posttrans_layers).model)

        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, g, h, e, snorm_n, snorm_e):
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            h = h_t

        h_out = self.MLP_layer(h)
        return h_out
    
    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
