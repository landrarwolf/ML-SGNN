import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from semantic import diffusion_fun_improved_ppmi_dynamic_sparsity
from sklearn.metrics.pairwise import pairwise_kernels
from utils import normalize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KERNELS = {
    1: ("rbf", {"gamma": 0.5}),
    2: ("cosine", {}),
    3: ("sigmoid", {}),
}


def parse_index_file(filename):
    """Read one integer node index per line."""
    with open(filename) as index_file:
        return [int(line.strip()) for line in index_file]


def process_planetoid_data(dataset):
    """Convert a Planetoid-format dataset into the ML-SGNN text format."""
    names = ("y", "ty", "ally", "x", "tx", "allx", "graph")
    objects = []
    cache_dir = DATA_DIR / "cache"

    for name in names:
        path = cache_dir / "ind.{}.{}".format(dataset, name)
        with path.open("rb") as data_file:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(data_file, encoding="latin1"))
            else:
                objects.append(pickle.load(data_file))

    y, ty, ally, x, tx, allx, graph = objects
    test_index_path = cache_dir / "ind.{}.test.index".format(dataset)
    test_indices_reordered = parse_index_file(test_index_path)
    test_indices = np.sort(test_indices_reordered)

    if dataset == "citeseer":
        full_range = range(min(test_indices_reordered), max(test_indices_reordered) + 1)
        extended_features = sp.lil_matrix((len(full_range), x.shape[1]))
        extended_features[test_indices - min(test_indices), :] = tx
        tx = extended_features
        extended_labels = np.zeros((len(full_range), y.shape[1]))
        extended_labels[test_indices - min(test_indices), :] = ty
        ty = extended_labels

    labels = np.vstack((ally, ty))
    labels[test_indices_reordered, :] = labels[test_indices, :]
    features = sp.vstack((allx, tx)).tolil()
    features[test_indices_reordered, :] = features[test_indices, :]
    features = features.toarray()

    dataset_dir = DATA_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    adjacency_path = dataset_dir / "{}.adj".format(dataset)
    with adjacency_path.open("w") as adjacency_file:
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                adjacency_file.write("{}\t{}\n".format(node, neighbor))

    label_ids = np.argmax(labels, axis=1)
    np.savetxt(dataset_dir / "{}.label".format(dataset), label_ids, fmt="%d")
    np.savetxt(dataset_dir / "{}.test".format(dataset), test_indices, fmt="%d")
    np.savetxt(dataset_dir / "{}.feature".format(dataset), features, fmt="%f")


def construct_feature_graph(dataset, features, top_k, measure):
    """Construct a k-nearest-neighbor graph using one similarity measure."""
    try:
        metric, keyword_arguments = KERNELS[measure]
    except KeyError:
        raise ValueError("measure must be one of {}".format(sorted(KERNELS)))

    similarities = pairwise_kernels(features, metric=metric, **keyword_arguments)
    neighbors = [
        np.argpartition(row, -(top_k + 1))[-(top_k + 1) :] for row in similarities
    ]

    output_dir = DATA_DIR / dataset / "knn{}".format(measure)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tmp.txt"
    with output_path.open("w") as output_file:
        for node, node_neighbors in enumerate(neighbors):
            for neighbor in node_neighbors:
                if neighbor != node:
                    output_file.write("{} {}\n".format(node, neighbor))
    return output_path


def generate_knn_graphs(dataset, measure):
    """Generate undirected k-NN graphs for k=2,...,9."""
    dataset_dir = DATA_DIR / dataset
    features = np.loadtxt(dataset_dir / "{}.feature".format(dataset), dtype=float)

    for top_k in range(2, 10):
        temporary_path = construct_feature_graph(dataset, features, top_k, measure)
        output_path = dataset_dir / "knn{}".format(measure) / "c{}.txt".format(top_k)
        with temporary_path.open("r") as input_file, output_path.open(
            "w"
        ) as output_file:
            for line in input_file:
                start, end = line.rstrip("\n").split(" ")
                if int(start) < int(end):
                    output_file.write("{} {}\n".format(start, end))


def construct_ppmi(dataset):
    """Generate and save the normalized semantic PPMI graph."""
    dataset_dir = DATA_DIR / dataset
    edge_path = dataset_dir / "{}.edge".format(dataset)
    output_path = dataset_dir / "ppmi.npz"

    structure_edges = np.genfromtxt(edge_path, dtype=np.int32)
    structure_edges = np.atleast_2d(structure_edges)
    adjacency = sp.coo_matrix(
        (
            np.ones(structure_edges.shape[0]),
            (structure_edges[:, 0], structure_edges[:, 1]),
        ),
        shape=(structure_edges.max() + 1, structure_edges.max() + 1),
        dtype=np.float32,
    )
    adjacency = (
        adjacency
        + adjacency.T.multiply(adjacency.T > adjacency)
        - adjacency.multiply(adjacency.T > adjacency)
    )
    normalized_adjacency = normalize(adjacency + sp.eye(adjacency.shape[0]))
    ppmi = diffusion_fun_improved_ppmi_dynamic_sparsity(
        normalized_adjacency, path_len=2, k=2.0
    )
    sp.save_npz(str(output_path), ppmi, compressed=True)
    print("Saved semantic graph to {}".format(output_path))


# Backward-compatible names from the original preprocessing script.
process_data = process_planetoid_data
construct_graph = construct_feature_graph
generate_knn = generate_knn_graphs


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare data for ML-SGNN.")
    parser.add_argument("--dataset", required=True, help="extracted dataset name")
    parser.add_argument(
        "--generate-knn",
        action="store_true",
        help="also generate all three families of feature k-NN graphs",
    )
    return parser


def main():
    args = build_parser().parse_args()
    construct_ppmi(args.dataset)
    if args.generate_knn:
        for measure in sorted(KERNELS):
            generate_knn_graphs(args.dataset, measure)


if __name__ == "__main__":
    main()
