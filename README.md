# ML-SGNN

[![Paper](https://img.shields.io/badge/paper-Engineering%20Applications%20of%20AI-blue)](https://doi.org/10.1016/j.engappai.2024.109647)
[![arXiv](https://img.shields.io/badge/arXiv-2212.01749-b31b1b.svg)](https://arxiv.org/abs/2212.01749)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official PyTorch implementation of **ML-SGNN**, introduced in
[*Semantic Graph Neural Network with Multi-measure Learning for Semi-supervised Classification*](https://doi.org/10.1016/j.engappai.2024.109647).

ML-SGNN learns node representations from three complementary graph views:

- a feature graph fused from RBF, cosine, and sigmoid similarities;
- the observed topology graph; and
- a semantic graph derived from random-walk co-occurrence statistics.

Attention layers combine the feature views and then aggregate the feature, topology, and semantic embeddings for semi-supervised node classification.

## Repository layout

```text
.
├── ML-SGNN/
│   ├── main.py             # Training and evaluation entry point
│   ├── models.py           # ML-SGNN model and attention modules
│   ├── layers.py           # Graph convolution layer
│   ├── utils.py            # Data and graph loading utilities
│   ├── semantic.py         # Semantic graph / PPMI utilities
│   ├── data_processing.py  # Dataset preprocessing helpers
│   └── config/             # Per-dataset experiment configurations
├── data/                   # Compressed benchmark datasets
├── CITATION.cff
└── requirements.txt
```

## Requirements

The original experiments used this legacy environment:

- Python 3.7
- PyTorch 1.1.0
- NumPy 1.16.2
- SciPy 1.3.1
- scikit-learn 0.21.3
- NetworkX 2.4

Create an isolated environment before installing the pinned dependencies:

```bash
python3.7 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Newer library versions may require small compatibility changes and have not been verified against the reported results.

## Data preparation

Six dataset archives are provided in `data/`:

| Dataset in the paper | Archive / CLI name |
| --- | --- |
| Citeseer | `citeseer` |
| UAI2010 | `uai` |
| ACM | `acm` |
| BlogCatalog | `BlogCatalog` |
| Flickr | `flickr` |
| CoraFull | `coraml` |

Extract the archives before training. For example:

```bash
unzip data/citeseer.zip -d data
```

Each extracted dataset follows this structure:

```text
citeseer/
├── citeseer.edge
├── citeseer.feature
├── citeseer.label
├── train20.txt
├── train40.txt
├── train60.txt
├── test20.txt
├── test40.txt
├── test60.txt
└── knn/
    ├── c2.txt
    └── ...
```

Generate the semantic PPMI graph when it is not already present:

```bash
cd ML-SGNN
python data_processing.py --dataset citeseer
```

To regenerate all three feature-graph families as well, add `--generate-knn`. This step computes pairwise similarities and can be expensive on larger datasets.

### Data sources

- **Citeseer:** [Semi-Supervised Classification with Graph Convolutional Networks](https://github.com/tkipf/gcn)
- **UAI2010:** *A Unified Weakly Supervised Framework for Community Detection and Semantic Matching*
- **ACM:** [Heterogeneous Graph Attention Network](https://github.com/Jhy1993/HAN)
- **BlogCatalog and Flickr:** [Co-Embedding Attributed Networks](https://github.com/mengzaiqiao/CAN)
- **CoraFull:** [Deep Gaussian Embedding of Graphs](https://github.com/abojchevski/graph2gauss)

## Configuration

Experiment settings are read from INI files in `ML-SGNN/config/`. By convention, the filename is `<labels-per-class><dataset>.ini`, for example `20citeseer.ini`. You can also pass a configuration file explicitly with `--config`.

> [!IMPORTANT]
> The current public release does not contain the original per-dataset INI files. The expected schema is documented in [`ML-SGNN/config/README.md`](ML-SGNN/config/README.md). Add the original experiment configurations before attempting to reproduce the paper's reported numbers.

## Training

Run commands from the `ML-SGNN` directory:

```bash
cd ML-SGNN
python main.py --dataset citeseer --label-rate 20
```

Available command-line options:

| Option | Description | Default |
| --- | --- | --- |
| `-d`, `--dataset` | Dataset/archive name from the table above | `citeseer` |
| `-l`, `--label-rate` | Number of labeled nodes per class | `20` |
| `-c`, `--config` | Explicit path to an INI configuration | inferred |

The training script reports loss, training accuracy, test accuracy, and macro-F1 at every epoch, followed by the best test result.

## Citation

If this code or paper is useful in your work, please cite:

```bibtex
@article{lin2025semantic,
  title   = {Semantic Graph Neural Network with Multi-measure Learning for Semi-supervised Classification},
  author  = {Lin, Junchao and Wan, Yuan and Xu, Jingwen and Qi, Xingchen},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {140},
  pages   = {109647},
  year    = {2025},
  doi     = {10.1016/j.engappai.2024.109647}
}
```

## License

This project is released under the [MIT License](LICENSE).
