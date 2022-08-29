import numpy as np
import scipy.sparse as sp
import json
import dataclasses
import random
import os
import networkx as nx

from typing import Sequence
# from training_info import *


@dataclasses.dataclass(frozen=True)
class GraphData:
    features: Sequence[np.ndarray]
    labels: Sequence[np.ndarray]
    adjacencies: Sequence[sp.csr_matrix]


@dataclasses.dataclass(frozen=True)
class InputData:
    features: Sequence[np.ndarray]
    labels: Sequence[np.ndarray]
    rows_1: Sequence[sp.csr_matrix]
    columns_1: Sequence[sp.csr_matrix]
    rows_2: Sequence[sp.csr_matrix]
    columns_2: Sequence[sp.csr_matrix]
    root_nodes: Sequence[int]


def batch(features, rows_1, cols_1, rows_2, cols_2, ys, root_nodes,max_features=None):
    """Converts a list of training examples into a batched single graph."""
    batch_size = len(features)
    if max_features is None:
        max_features = max(f.shape[0] for f in features)
    b_features = np.zeros((batch_size, max_features, features[0].shape[1]))
    b_rows_1 = []
    b_cols_1 = []
    b_rows_2 = []
    b_cols_2 = []
    b_ys = ys
    b_masks = np.zeros((batch_size, max_features, 1))
    # b_masks = np.zeros((batch_size, 1, max_features))
    for i in range(batch_size):
        b_features[i, :features[i].shape[0], :] = features[i]
        b_rows_1.append(rows_1[i] + i * max_features)
        b_cols_1.append(cols_1[i] + i * max_features)
        b_rows_2.append(rows_2[i] + i * max_features)
        b_cols_2.append(cols_2[i] + i * max_features)
        # b_ys[i, 0] = ys[i, 0]
        root_node = root_nodes[i]
        b_masks[i,:features[i].shape[0],0] = 1
        # print(features[i].shape[0])
    # print(b_masks)
    b_features = b_features.reshape((-1, b_features.shape[-1]))
    b_rows_1 = np.concatenate(b_rows_1)
    b_cols_1 = np.concatenate(b_cols_1)
    b_rows_2 = np.concatenate(b_rows_2)
    b_cols_2 = np.concatenate(b_cols_2)

    return b_features, b_rows_1, b_cols_1, b_rows_2, b_cols_2, b_ys, b_masks


def convert_networkx_to_adjacency_input(graph):
    adjacency_matrix = nx.to_scipy_sparse_array(graph, format='coo')
    adjacency_matrix += sp.eye(adjacency_matrix.shape[0])
    return adjacency_matrix


def read_partition_multiplicity(GRAPH_DIR, N):
    with open(os.path.join(GRAPH_DIR, f"XP_{N}_multiplicity.json")) as f:
        return json.load(f)


def iter_graph(GRAPH_DIR, NUM_GRAPHS):
    for i in range(NUM_GRAPHS):
        filename = os.path.join(GRAPH_DIR, f"graph_{i:05d}.npz")
        yield nx.from_scipy_sparse_array(
            sp.load_npz(filename), create_using=nx.DiGraph)


def generate_graph_data(N, partition_parts, feat_list, GRAPH_DIR, NUM_GRAPHS):
    """Generate dataset for training GraphNet model on KL data.

    This generates a dataset for training a GraphNet model.

    Args:
    partition_parts

    Returns:
    An GraphData instance with features, adjacencies and labels.
    """
    par_mults = read_partition_multiplicity(GRAPH_DIR,N)

    ys = np.array([par_mult for par_mult in par_mults])
    ys = ys[:, [part-1 for part in partition_parts]]

    features = []
    adjacencies = []

    for graph in iter_graph(GRAPH_DIR, NUM_GRAPHS):
        feat_dict = dict()
        for key, feat in feat_list.items():
            feat_dict[key] = feat(graph)

        curr_feature = np.zeros((len(graph), len(feat_dict)))

        for n, perm in enumerate(graph.nodes):
            for i, (name, value) in enumerate(feat_dict.items()):
                curr_feature[n, i] = value[perm]

        features.append(curr_feature)
        adjacencies.append(convert_networkx_to_adjacency_input(graph))

    return GraphData(features=features, labels=ys, adjacencies=adjacencies)


def get_root_node(col):
    return np.bincount(col).argmin()


def load_input_data(train_fraction, GRAPH_DIR, NUM_GRAPHS, N, partition_parts=None, feat_list=None, extended=True, label_size=None):
    """Loads input data for the specified prediction problem.

    This loads a dataset that can be used with a GraphNet model. The Bruhat
    intervals are taken from the dataset of intervals in S9 and the label
    is the coefficient of specified degree.

    The datasets are cached, and only regenerated when not found on disk.

    Args:
    partition_parts: the parts to use as the label.
    extended: True if training data to be extended
    Returns:
    Three InputData instances with features, rows, cols and labels. They are the
    full/train/test set respectively.
    """

    if type(partition_parts) == int:
        partition_parts = [partition_parts]
    if partition_parts == None:
        partition_parts = [i for i in range(1,N+1)]

    print(f"Generating data for partition_parts {partition_parts}", flush=True)
    graph_data = generate_graph_data(N, partition_parts, feat_list, GRAPH_DIR, NUM_GRAPHS)
    features = graph_data.features
    adjacencies = graph_data.adjacencies
    ys = graph_data.labels

    # zip_data = list(
    #     zip(
    #         graph_data.features,
    #         graph_data.adjacencies,
    #         graph_data.labels,
    #     ))
    # random.shuffle(zip_data)
    # features, adjacencies, ys = zip(*zip_data)
    # features = list(features)
    # adjacencies = list(adjacencies)
    # ys = np.array(ys)

    num_training = int(len(ys) * train_fraction)
    num_testing = int(len(ys) * (1 - train_fraction))
    max_label_size = np.max(np.array(label_size[N])[partition_parts])

    if extended:
        data_n = len(ys)
        # zero_pos = list(np.where(ys[num_testing:] == 0)[0])
        # nonzero_pos = list(np.where(ys[num_testing:] != 0)[0])
        # for p in zero_pos:
        #     p += num_testing
        #     p_feature = features[p]
        #     p_adjacencies = [sp.coo_matrix(adjacencies[p])]
        #     p_y = np.array(ys[p])
        #     for i in range((p % 2) + 1):
        #         q = random.choice(nonzero_pos) + num_testing
        #         p_feature = np.append(p_feature, features[q], axis=0)
        #         p_adjacencies.append(sp.coo_matrix(adjacencies[q]))
        #         p_y += ys[q]
        #     if p % 5 == 0:
        #         q = np.random.randint(num_testing, data_n)
        #         p_feature = np.append(p_feature, features[q], axis=0)
        #         p_adjacencies.append(sp.coo_matrix(adjacencies[q]))
        #         p_y += ys[q]
        #     if label_size is None or np.max(p_y) <= max_label_size:
        #         features.append(p_feature)
        #         adjacencies.append(sp.csr_array(sp.block_diag(p_adjacencies)))
        #         ys = np.append(ys, p_y.reshape(-1, len(p_y)), axis=0)
        for i in range(num_training):
            p = np.random.randint(num_testing, data_n)
            q = np.random.randint(num_testing, data_n)
            append_feat = features[p]
            append_feat = np.append(append_feat, features[q], axis=0)
            append_adj = [sp.coo_matrix(adjacencies[p]), sp.coo_matrix(adjacencies[q])]
            append_ys = np.array(ys[p])
            append_ys += ys[q]
            if label_size is None or np.max(append_ys) <= max_label_size:
                features.append(append_feat)
                adjacencies.append(sp.csr_array(sp.block_diag(append_adj)))
                ys = np.append(ys, append_ys.reshape(-1, len(append_ys)), axis=0)


    rows = [sp.coo_matrix(a).row for a in adjacencies]
    cols = [sp.coo_matrix(a).col for a in adjacencies]
    rows_1 = []
    cols_1 = []
    rows_2 = []
    cols_2 = []
    for i in range(len(rows)):
        rows_1.append(np.array(rows[i], dtype=np.int16))
        cols_1.append(np.array(cols[i], dtype=np.int16))
        Hasse_rows, Hasse_cols = Hasse_diagram(rows[i], cols[i])
        Hasse_rows, Hasse_cols = go_right(Hasse_rows, Hasse_cols)
        rows_2.append(Hasse_rows)
        cols_2.append(Hasse_cols)
    root_nodes = [get_root_node(col) for col in cols]

    features_test = features[:num_testing]
    rows_1_test = [row for row in rows_1[:num_testing]]
    cols_1_test = [col for col in cols_1[:num_testing]]
    rows_2_test = [row for row in rows_2[:num_testing]]
    cols_2_test = [col for col in cols_2[:num_testing]]
    ys_test = ys[:num_testing]
    root_nodes_test = root_nodes[:num_testing]

    features_train = features[num_testing:]
    rows_1_train = [row for row in rows_1[num_testing:]]
    cols_1_train = [col for col in cols_1[num_testing:]]
    rows_2_train = [row for row in rows_2[num_testing:]]
    cols_2_train = [col for col in cols_2[num_testing:]]
    ys_train = ys[num_testing:]
    root_nodes_train = root_nodes[num_testing:]

    return (
        InputData(features=features, rows_1=rows_1, columns_1=cols_1, rows_2=rows_2, columns_2=cols_2, labels=ys,
                  root_nodes=root_nodes),
        InputData(features=features_train, rows_1=rows_1_train, columns_1=cols_1_train, rows_2=rows_2_train,
                  columns_2=cols_2_train, labels=ys_train, root_nodes=root_nodes_train),
        InputData(features=features_test, rows_1=rows_1_test, columns_1=cols_1_test, rows_2=rows_2_test,
                  columns_2=cols_2_test, labels=ys_test, root_nodes=root_nodes_test))

def Hasse_diagram(rows, cols):
    Hasse_rows = []
    Hasse_cols = []
    for i in range(len(rows)):
        if rows[i] == cols[i]:
            Hasse_rows.append(rows[i])
            Hasse_cols.append(cols[i])
            continue
        chk = 0
        for j in list(np.where(rows==rows[i])[0]):
            if j == i or rows[j] == cols[j]:
                continue
            for k in list(np.where(cols==cols[i])[0]):
                if k == i or rows[k] == cols[k]:
                    continue
                if cols[j] == rows[k]:
                    chk = 1
                    break
            if chk == 1:
                break
        if chk == 0:
            Hasse_rows.append(rows[i])
            Hasse_cols.append(cols[i])
    return np.array(Hasse_rows, dtype=np.int16), np.array(Hasse_cols, dtype=np.int16)

def go_left(Hasse_rows, Hasse_cols):
    bincnt = np.bincount(Hasse_rows)
    left_rows = []
    left_cols = []
    for i in range(max(Hasse_rows)+1):
        for p in list(np.where(Hasse_rows==i)[0]):
            if bincnt[i] < 3 or Hasse_cols[p] <= i:
                left_rows.append(Hasse_rows[p])
                left_cols.append(Hasse_cols[p])
    return np.array(left_rows, dtype=np.int16), np.array(left_cols, dtype=np.int16)

def go_right(Hasse_rows, Hasse_cols):
    bincnt = np.bincount(Hasse_rows)
    right_rows = []
    right_cols = []
    for i in range(max(Hasse_rows)+1):
        for p in list(np.where(Hasse_rows==i)[0]):
            if bincnt[i] < 3 or Hasse_cols[p] >= i:
                right_rows.append(Hasse_rows[p])
                right_cols.append(Hasse_cols[p])
    return np.array(right_rows, dtype=np.int16), np.array(right_cols, dtype=np.int16)
