import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # Relu(AXW)
        x = F.dropout(x, self.dropout, training=self.training)  # dropout
        x = self.gc2(x, adj)  # AXW
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    # hidden_size == 16

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class GMA_GCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(GMA_GCN, self).__init__()

        self.F1 = GCN(nfeat, nhid1, nhid2, dropout)  # feature graph 1 nfeat: 3703 -> 768 -> 256
        self.F2 = GCN(nfeat, nhid1, nhid2, dropout)  # feature graph 2
        self.F3 = GCN(nfeat, nhid1, nhid2, dropout)  # feature graph 3
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)  # Structure graph
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)  # Structure graph
        self.SEM = GCN(nfeat, nhid1, nhid2, dropout)  # Semantic graph
        self.SEM2 = GCN(nfeat, nhid1, nhid2, dropout)  # Semantic graph
        self.dropout = dropout

        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(nhid2, 16)

        self.b = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.attention_all = Attention(nhid2, 32)

        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, sadj, fadj1, fadj2, fadj3, ppmi):
        # Feature graph
        f1 = self.F1(x, fadj1)  # feature graph 1
        f2 = self.F2(x, fadj2)  # feature graph 2
        f3 = self.F3(x, fadj3)  # feature graph 3

        fadj = torch.stack([f1, f2, f3], dim=1)
        fadj, att_f = self.attention(fadj)

        # Structure graph
        str = self.SGCN(x, sadj)

        # Semantic graph
        sem = self.SEM(x, ppmi)

        emb = torch.stack([fadj, str, sem], dim=1)
        emb, att_all = self.attention_all(emb)

        output = self.MLP(emb)

        return output, fadj, str, sem, emb
