import torch
import torch.nn as nn

class JKNet(nn.Module):
    """
    Jumping Knowledge Networks 
    - flexible Conv operator and
    - works with both 'dgl' and 'geometric' backends.
    """

    def __init__(self, conv_cls, in_feats, n_hidden, n_classes, n_layers, activation, dropout,
            mode='cat', conv_args=None, conv_kwargs=None, backend='dgl'):
        super().__init__()
        conv_args = conv_args if conv_args is not None else []
        conv_kwargs = conv_kwargs if conv_kwargs is not None else {}
        self.layers = nn.ModuleList()
        self.layers.append(conv_cls(in_feats, n_hidden, *conv_args, **conv_kwargs))
        assert n_layers > 1, "JKNet with <2 conv layers does not make much sense"
        for __ in range(n_layers - 1):
            self.layers.append(conv_cls(n_hidden, n_hidden, *conv_args, **conv_kwargs))
        self.output_layer = nn.Linear(n_layers * n_hidden, n_classes)

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        self.reset_parameters()
        assert backend in ['geometric', 'dgl']
        self.backend = backend

    def forward(self, *args, **kwargs):
        """
        Args are expected as g, x if backend is dgl, else x, edge_index.
        Keyword args (such as edge_weight) will be forwarded.
        """
        if self.backend == 'dgl':
            # Trainer supplies swapped args for dgl
            g, x = args
        elif self.backend == 'geometric':
            x, edge_index = args
        else:
            raise ValueError("Unknown backend:" + self.backend)


        activations = []
        for layer in self.layers:
            if self.backend == 'dgl':
                x = layer(g, x, **kwargs)
            else:
                x = layer(x, edge_index, **kwargs)
            x = self.dropout(self.activation(x))
            activations.append(x)
        if self.mode == 'cat':
            h = torch.cat(activations, dim=1)
        else:
            raise NotImplementedError("[jknet] Aggregation Mode not implemented:" + self.mode)
        return self.output_layer(h)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.output_layer.reset_parameters()

    def reset_final_parameters(self):
        self.output_layer.reset_parameters()

    def final_parameters(self):
        yield self.output_layer.weight
        yield self.output_layer.bias
