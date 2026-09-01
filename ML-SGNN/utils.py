import numpy as np
import scipy.sparse as sp
import torch


def common_loss(embedding_1, embedding_2):
    """Measure disagreement between two centered embedding Gram matrices."""
    embedding_1 = embedding_1 - torch.mean(embedding_1, dim=0, keepdim=True)
    embedding_2 = embedding_2 - torch.mean(embedding_2, dim=0, keepdim=True)
    embedding_1 = torch.nn.functional.normalize(embedding_1, p=2, dim=1)
    embedding_2 = torch.nn.functional.normalize(embedding_2, p=2, dim=1)
    covariance_1 = torch.matmul(embedding_1, embedding_1.t())
    covariance_2 = torch.matmul(embedding_2, embedding_2.t())
    return torch.mean((covariance_1 - covariance_2) ** 2)


def accuracy(output, labels):
    predictions = output.max(dim=1)[1].type_as(labels)
    correct = predictions.eq(labels).double().sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_matrix):
    """Convert a SciPy sparse matrix to a PyTorch sparse tensor."""
    sparse_matrix = sparse_matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_matrix.row, sparse_matrix.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_matrix.data)
    shape = torch.Size(sparse_matrix.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Read one integer node index per line."""
    with open(filename) as index_file:
        return [int(line.strip()) for line in index_file]


def sample_mask(indices, length):
    """Create a Boolean mask from a collection of indices."""
    mask = np.zeros(length)
    mask[indices] = 1
    return np.asarray(mask, dtype=bool)


def sparse_to_tuple(sparse_matrix):
    """Convert a sparse matrix, or list of matrices, to tuple form."""

    def to_tuple(matrix):
        if not sp.isspmatrix_coo(matrix):
            matrix = matrix.tocoo()
        coordinates = np.vstack((matrix.row, matrix.col)).transpose()
        return coordinates, matrix.data, matrix.shape

    if isinstance(sparse_matrix, list):
        return [to_tuple(matrix) for matrix in sparse_matrix]
    return to_tuple(sparse_matrix)


def normalize(matrix):
    """Row-normalize a sparse matrix."""
    row_sum = np.asarray(matrix.sum(axis=1))
    inverse_row_sum = np.power(row_sum, -1).flatten()
    inverse_row_sum[np.isinf(inverse_row_sum)] = 0.0
    return sp.diags(inverse_row_sum).dot(matrix)


def load_data(config):
    feature_array = np.loadtxt(config.feature_path, dtype=float)
    label_array = np.loadtxt(config.label_path, dtype=int)
    test_indices = np.loadtxt(config.test_path, dtype=int)
    train_indices = np.loadtxt(config.train_path, dtype=int)

    feature_matrix = sp.csr_matrix(feature_array, dtype=np.float32)
    features = torch.FloatTensor(np.asarray(feature_matrix.todense()))
    labels = torch.LongTensor(np.asarray(label_array))
    train_indices = torch.LongTensor(train_indices.tolist())
    test_indices = torch.LongTensor(test_indices.tolist())

    return features, labels, train_indices, test_indices


def _load_normalized_adjacency(edge_path, node_count):
    edges = np.genfromtxt(edge_path, dtype=np.int32)
    edges = np.atleast_2d(edges)
    adjacency = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(node_count, node_count),
        dtype=np.float32,
    )
    adjacency = (
        adjacency
        + adjacency.T.multiply(adjacency.T > adjacency)
        - adjacency.multiply(adjacency.T > adjacency)
    )
    return normalize(adjacency + sp.eye(adjacency.shape[0]))


def load_graph(_dataset, config):
    """Load the topology, three feature graphs, and semantic graph."""
    feature_paths = (
        config.featuregraph_path_1 + str(config.k) + ".txt",
        config.featuregraph_path_2 + str(config.k) + ".txt",
        config.featuregraph_path_3 + str(config.k) + ".txt",
    )
    feature_adjacencies = [
        _load_normalized_adjacency(path, config.n) for path in feature_paths
    ]
    structure_adjacency = _load_normalized_adjacency(config.structgraph_path, config.n)
    semantic_adjacency = sp.load_npz(config.ppmi_path)

    return (
        sparse_mx_to_torch_sparse_tensor(structure_adjacency),
        sparse_mx_to_torch_sparse_tensor(feature_adjacencies[0]),
        sparse_mx_to_torch_sparse_tensor(feature_adjacencies[1]),
        sparse_mx_to_torch_sparse_tensor(feature_adjacencies[2]),
        sparse_mx_to_torch_sparse_tensor(semantic_adjacency),
    )
