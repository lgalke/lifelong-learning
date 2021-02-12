"""
All network architectures based on the torch geometric framework are here
https://github.com/rusty1s/pytorch_geometric
These are meant to be used with torch-geometric's GraphSAINT sampler
"""
from abc import ABC

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GraphConv, GATConv, SGConv, GCNConv
from torch_geometric.transforms import SIGN as SIGN_Trans
import torch.nn.functional as F


class _BasicNet(nn.Module, ABC):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 layer, **layer_kwargs):
        super(_BasicNet, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        # print("in_feats", in_feats)
        # print("TYPE of in_feats", type(in_feats))
        self.layers.append(layer(in_feats, n_hidden, **layer_kwargs))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(layer(n_hidden, n_hidden, **layer_kwargs))
        # output layer
        self.layers.append(layer(n_hidden, n_classes, **layer_kwargs))
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, edge_index, edge_norm=None):
        h = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GATConv):
                h = layer(x=h, edge_index=edge_index, edge_weight=edge_norm)
            else:
                h = layer(x=h, edge_index=edge_index)
            if i < len(self.layers) - 1:
                h = self.activation(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def reset_final_parameters(self):
        self.layers[-1].reset_parameters()

    def final_parameters(self):
        pass


class GraphSAGE(_BasicNet):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GraphSAGE, self).__init__(in_feats, n_hidden, n_classes, n_layers, activation, dropout, SAGEConv)

    def final_parameters(self):
        yield self.layers[-1].lin_rel.weight
        yield self.layers[-1].lin_rel.bias
        yield self.layers[-1].lin_root.weight


class GCN(_BasicNet):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 improved=False):
        super(GCN, self).__init__(in_feats, n_hidden, n_classes, n_layers, activation, dropout, GCNConv, improved=improved)

    def final_parameters(self):
        yield self.layers[-1].lin.weight
        yield self.layers[-1].lin.bias
        yield self.layers[-1].weight


class GAT(_BasicNet):
    def __init__(self,
                 in_feats,
                 n_hidden_per_head,
                 n_classes,
                 activation,
                 dropout,
                 attn_dropout,
                 heads):
        # sense-free call to superclass
        super(GAT, self).__init__(1, 1, 1, 1, 1, 1, GraphConv)

        # now override the network architecture, because GAT is special
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_feats, n_hidden_per_head, heads=heads[0], dropout=attn_dropout))
        # hidden layers
        self.layers.append(GATConv(n_hidden_per_head * heads[0], n_classes, heads=heads[1], concat=True,
                                   dropout=attn_dropout))
        # output layer
        self.activation = activation
        self.dropout = dropout

    def final_parameters(self):
        yield self.layers[-1].lin.weight
        yield self.layers[-1].bias


class MLP(_BasicNet):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(MLP, self).__init__(in_feats=in_feats, n_hidden=n_hidden, n_classes=n_classes, n_layers=n_layers,
                                  activation=activation, dropout=dropout, layer=nn.Linear)

    def forward(self, x, edge_index, edge_norm=None):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def final_parameters(self):
        yield self.layers[-1].weight
        yield self.layers[-1].bias




class SGNet(SGConv):
    def __reset_cache__(self):
        self._cached_x = None

    def final_parameters(self):
        yield self.lin.weight
        yield self.lin.bias

    def reset_final_parameters(self):
        self.lin.reset_parameters()
