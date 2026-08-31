import math

import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """Graph convolution layer from Kipf and Welling (2017)."""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, features, adjacency):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return "{} ({} -> {})".format(
            self.__class__.__name__, self.in_features, self.out_features
        )
