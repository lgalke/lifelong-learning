"""
GCN implementation of the DGL library (https://dgl.ai) with minor modifications
to facilitate dynamically changing graph structure.

Source: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/gcn_mp.py
"""
import math
import torch
import torch.nn as nn

# In DGL, GraphConv refers to Kipf's GCN Conv operator
from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 improved=True):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def reset_final_parameters(self):
        self.layers[-1].reset_parameters()

    def final_parameters(self):
        yield self.layers[-1].fc.weight
        yield self.layers[-1].fc.bias
