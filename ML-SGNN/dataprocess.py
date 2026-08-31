"""Backward-compatible wrapper for the renamed data_processing module."""

from data_processing import (
    construct_feature_graph,
    construct_graph,
    construct_ppmi,
    generate_knn,
    generate_knn_graphs,
    main,
    parse_index_file,
    process_data,
    process_planetoid_data,
)

if __name__ == "__main__":
    main()
