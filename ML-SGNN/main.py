import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from config import Config
from models import MLSGNN
from sklearn.metrics import f1_score
from utils import accuracy, common_loss, load_data, load_graph

PROJECT_DIR = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train ML-SGNN for semi-supervised node classification."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="citeseer",
        help="dataset name used by the configuration file (default: citeseer)",
    )
    parser.add_argument(
        "-l",
        "--label-rate",
        "--labelrate",
        dest="label_rate",
        type=int,
        default=20,
        help="number of labeled training nodes per class (default: 20)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="path to an INI configuration file; inferred when omitted",
    )
    return parser


def resolve_config_path(args):
    if args.config is not None:
        return args.config
    filename = "{}{}.ini".format(args.label_rate, args.dataset)
    return PROJECT_DIR / "config" / filename


def set_random_seed(seed, use_cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model,
    features,
    labels,
    test_indices,
    structure_adjacency,
    feature_adjacencies,
    semantic_adjacency,
):
    model.eval()
    with torch.no_grad():
        output, _, _, _, embedding = model(
            features,
            structure_adjacency,
            feature_adjacencies[0],
            feature_adjacencies[1],
            feature_adjacencies[2],
            semantic_adjacency,
        )

    test_accuracy = accuracy(output[test_indices], labels[test_indices])
    predictions = output[test_indices].argmax(dim=1).cpu().numpy()
    test_labels = labels[test_indices].cpu().numpy()
    macro_f1 = f1_score(test_labels, predictions, average="macro")
    return test_accuracy.item(), float(macro_f1), embedding


def train_epoch(
    model,
    optimizer,
    config,
    epoch,
    features,
    labels,
    train_indices,
    test_indices,
    structure_adjacency,
    feature_adjacencies,
    semantic_adjacency,
):
    model.train()
    optimizer.zero_grad()

    output, feature_embedding, topology_embedding, semantic_embedding, _ = model(
        features,
        structure_adjacency,
        feature_adjacencies[0],
        feature_adjacencies[1],
        feature_adjacencies[2],
        semantic_adjacency,
    )

    classification_loss = F.nll_loss(output[train_indices], labels[train_indices])
    if config.beta == 0 and config.theta == 0:
        loss = classification_loss
    else:
        topology_semantic_loss = common_loss(topology_embedding, semantic_embedding)
        topology_feature_loss = common_loss(topology_embedding, feature_embedding)
        loss = (
            classification_loss
            + config.beta * topology_semantic_loss
            + config.theta * topology_feature_loss
        )

    train_accuracy = accuracy(output[train_indices], labels[train_indices])
    loss.backward()
    optimizer.step()

    test_accuracy, macro_f1, _ = evaluate(
        model,
        features,
        labels,
        test_indices,
        structure_adjacency,
        feature_adjacencies,
        semantic_adjacency,
    )

    print(
        "epoch:{:04d}".format(epoch),
        "loss:{:.4f}".format(loss.item()),
        "train_acc:{:.4f}".format(train_accuracy.item()),
        "test_acc:{:.4f}".format(test_accuracy),
        "test_macro_f1:{:.4f}".format(macro_f1),
    )
    return loss.item(), test_accuracy, macro_f1


def run(args):
    config = Config(resolve_config_path(args))
    use_cuda = not config.no_cuda and torch.cuda.is_available()

    if not config.no_seed:
        set_random_seed(config.seed, use_cuda)

    (
        structure_adjacency,
        feature_adjacency_1,
        feature_adjacency_2,
        feature_adjacency_3,
        semantic_adjacency,
    ) = load_graph(args.dataset, config)
    features, labels, train_indices, test_indices = load_data(config)
    feature_adjacencies = (
        feature_adjacency_1,
        feature_adjacency_2,
        feature_adjacency_3,
    )

    model = MLSGNN(
        nfeat=config.fdim,
        nhid1=config.nhid1,
        nhid2=config.nhid2,
        nclass=config.class_num,
        n=config.n,
        dropout=config.dropout,
    )

    if use_cuda:
        model = model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        train_indices = train_indices.cuda()
        test_indices = test_indices.cuda()
        structure_adjacency = structure_adjacency.cuda()
        feature_adjacencies = tuple(
            adjacency.cuda() for adjacency in feature_adjacencies
        )
        semantic_adjacency = semantic_adjacency.cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_accuracy = 0.0
    best_macro_f1 = 0.0
    best_epoch = 0

    for epoch in range(config.epochs):
        _, test_accuracy, macro_f1 = train_epoch(
            model,
            optimizer,
            config,
            epoch,
            features,
            labels,
            train_indices,
            test_indices,
            structure_adjacency,
            feature_adjacencies,
            semantic_adjacency,
        )
        if test_accuracy >= best_accuracy:
            best_accuracy = test_accuracy
            best_macro_f1 = macro_f1
            best_epoch = epoch

    print(
        "best_epoch:{}".format(best_epoch),
        "best_test_acc:{:.4f}".format(best_accuracy),
        "best_test_macro_f1:{:.4f}".format(best_macro_f1),
    )


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
