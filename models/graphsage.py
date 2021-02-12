"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE, derived from dgl/examples.
"""

import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, graph, features):
        h = features
        for layer in self.layers:
            h = layer(graph, h)
        return h

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def reset_final_parameters(self):
        self.layers[-1].reset_parameters()

    def final_parameters(self):
        yield self.layers[-1].fc_self.weight
        yield self.layers[-1].fc_self.bias
        yield self.layers[-1].fc_neigh.weight
        yield self.layers[-1].fc_neigh.bias


