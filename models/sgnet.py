""" Subclass of DGL's Simplified GCN Implementation to enable incremental training"""

from dgl.nn.pytorch.conv.sgconv import SGConv


class SGNet(SGConv):
    def __reset_cache__(self):
        self._cached_h = None

    def reset_final_parameters(self):
        # SGNet has only one input-to-output layer
        self.reset_parameters()

    def final_parameters(self):
        yield self.fc.weight
        yield self.fc.bias
