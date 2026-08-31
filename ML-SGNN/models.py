import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    """Two-layer graph convolutional encoder."""

    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, features, adjacency):
        hidden = F.relu(self.gc1(features, adjacency))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        return self.gc2(hidden, adjacency)


class ViewAttention(nn.Module):
    """Learn attention weights over multiple graph-view embeddings."""

    def __init__(self, in_size, hidden_size):
        super(ViewAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, embeddings):
        scores = self.project(embeddings)
        weights = torch.softmax(scores, dim=1)
        return (weights * embeddings).sum(dim=1), weights


class MLSGNN(nn.Module):
    """Multi-measure learning semantic graph neural network."""

    def __init__(self, nfeat, nclass, nhid1, nhid2, n, dropout):
        super(MLSGNN, self).__init__()
        self.node_count = n

        self.F1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.F2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.F3 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SEM = GCN(nfeat, nhid1, nhid2, dropout)
        self.SEM2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.dropout = dropout

        # These parameters and duplicate encoders are retained for compatibility
        # with checkpoints produced by the original code release.
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = ViewAttention(nhid2, 16)

        self.b = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.attention_all = ViewAttention(nhid2, 32)

        self.MLP = nn.Sequential(nn.Linear(nhid2, nclass), nn.LogSoftmax(dim=1))

    def forward(
        self,
        features,
        structure_adjacency,
        feature_adjacency_1,
        feature_adjacency_2,
        feature_adjacency_3,
        semantic_adjacency,
    ):
        feature_embedding_1 = self.F1(features, feature_adjacency_1)
        feature_embedding_2 = self.F2(features, feature_adjacency_2)
        feature_embedding_3 = self.F3(features, feature_adjacency_3)

        feature_embeddings = torch.stack(
            [feature_embedding_1, feature_embedding_2, feature_embedding_3], dim=1
        )
        feature_embedding, _ = self.attention(feature_embeddings)

        topology_embedding = self.SGCN(features, structure_adjacency)
        semantic_embedding = self.SEM(features, semantic_adjacency)

        graph_view_embeddings = torch.stack(
            [feature_embedding, topology_embedding, semantic_embedding], dim=1
        )
        combined_embedding, _ = self.attention_all(graph_view_embeddings)
        output = self.MLP(combined_embedding)

        return (
            output,
            feature_embedding,
            topology_embedding,
            semantic_embedding,
            combined_embedding,
        )


# Backward-compatible names used by the original release.
Attention = ViewAttention
GMA_GCN = MLSGNN
