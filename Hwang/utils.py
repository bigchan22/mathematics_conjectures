import warnings
warnings.filterwarnings('ignore')

import collections
import dataclasses
import datetime
import enum
import functools
import itertools
import json
import os
import pickle
import random
import tempfile
from typing import Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optax
import psutil
import scipy.sparse as sp
import seaborn as sns

train_fraction = .8

def random_feature(D):
    rand_feature = dict.fromkeys(D.nodes)
    for key in rand_feature.keys():
        rand_feature[key] = np.random.rand()
    return rand_feature

def constant_feature(D):
    const_feature = dict.fromkeys(D.nodes)
    for key in const_feature.keys():
        const_feature[key] = 1
    return const_feature

def numbering_feature(D):
    num_feature = dict()
    for n, node in enumerate(D.nodes):
        num_feature[node] = n
    return num_feature

def get_sinks(D):
    return (node for node, out_dg in D.out_degree() if out_dg == 0)


def shortest_path_lengths(D):
    sinks = get_sinks(D)
    shortest_lengths = dict.fromkeys(D.nodes, float('inf'))
    for sink in sinks:
        for node, length in nx.shortest_path_length(D, target=sink).items():
            if shortest_lengths[node] > length:
                shortest_lengths[node] = length
    return shortest_lengths


def longest_path_length_to_target(D, target):
    dist = dict.fromkeys(D.nodes, -float('inf'))
    dist[target] = 0
    topo_order = reversed(list(nx.topological_sort(D)))
    for v in topo_order:
        for u in D.predecessors(v):
            if dist[u] < dist[v] + 1:
                dist[u] = dist[v] + 1
    return dist


def longest_path_lengths(D):
    sinks = get_sinks(D)
    longest_lengths = dict.fromkeys(D.nodes, -float('inf'))
    for sink in sinks:
        for node, length in longest_path_length_to_target(D, target=sink).items():
            if longest_lengths[node] < length:
                longest_lengths[node] = length
    return longest_lengths


def convert_networkx_to_adjacency_input(graph):
    adjacency_matrix = nx.to_scipy_sparse_array(graph, format='coo')
    adjacency_matrix += sp.eye(adjacency_matrix.shape[0])
    return adjacency_matrix


@dataclasses.dataclass(frozen=True)
class GraphData:
    features: Sequence[np.ndarray]
    labels: Sequence[np.ndarray]
    adjacencies: Sequence[sp.csr_matrix]


def generate_graph_data(N, partition_part, feat_list):
    """Generate dataset for training GraphNet model on KL data.

    This generates a dataset for training a GraphNet model.

    Args:
    partition_part: The polynomial coefficient to use as the label.

    Returns:
    An GraphData instance with features, adjacencies and labels.
    """
    par_mults = read_partition_multiplicity(N)

    ys = np.array([par_mult for par_mult in par_mults])
    ys = ys[:, partition_part - 1:partition_part]

    features = []
    adjacencies = []

    for graph in iter_graph():
#         feat_dict = {
# #             'in_centrality': nx.in_degree_centrality(graph),
# #             'out_centrality': nx.out_degree_centrality(graph),
# #             'shortest_path_length': shortest_path_lengths(graph),
# #             'longest_path_lengths': longest_path_lengths(graph),
# #             'random_feature': random_feature(graph),
#             'constant_feature': constant_feature(graph),
# #             'numbering_feature': numbering_feature(graph),
#         }
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


def iter_graph():
    for i in range(NUM_GRAPHS):
        filename = os.path.join(GRAPH_DIR, f"graph_{i:05d}.npz")
        yield nx.from_scipy_sparse_array(
            sp.load_npz(filename), create_using=nx.DiGraph)


def get_root_node(col):
    return np.bincount(col).argmin()


def read_partition_multiplicity(N):
    with open(os.path.join(GRAPH_DIR, f"XP_{N}_multiplicity.json")) as f:
        return json.load(f)

@dataclasses.dataclass(frozen=True)
class InputData:
    features: Sequence[np.ndarray]
    labels: Sequence[np.ndarray]
    rows: Sequence[sp.csr_matrix]
    columns: Sequence[sp.csr_matrix]
    root_nodes: Sequence[int]


def load_input_data(N=7, partition_part=1, feat_list=None, extended=True, label_size=None):
    """Loads input data for the specified prediction problem.

    This loads a dataset that can be used with a GraphNet model. The Bruhat
    intervals are taken from the dataset of intervals in S9 and the label
    is the coefficient of specified degree.

    The datasets are cached, and only regenerated when not found on disk.

    Args:
    partition_part: the part to use as the label.
    extended: True if training data to be extended
    Returns:
    Three InputData instances with features, rows, cols and labels. They are the
    full/train/test set respectively.
    """

    print(f"Generating data for partition_part {partition_part}", flush=True)
    graph_data = generate_graph_data(N, partition_part, feat_list)
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

    
############################################################################
#     graph_data = generate_graph_data(partition_part)
# #     features = graph_data.features
# #     adjacencies = graph_data.adjacencies
#     features = []
#     adjacencies = []
#     ys = graph_data.labels
#     rand_indices = np.arange(len(ys))
#     np.random.shuffle(rand_indices)
#     for i in range(len(ys)):
#         features.append(graph_data.features[rand_indices[i]])
#         adjacencies.append(graph_data.adjacencies[rand_indices[i]])
# #     rand_indices = np.arange(len(ys))   # random labels
# #     np.random.shuffle(rand_indices)
#     ys = ys[rand_indices]
############################################################################
    
    num_training = int(len(ys) * train_fraction)
    num_testing = int(len(ys) * (1-train_fraction))
    
    if extended:
        data_n = len(ys)
        zero_pos = list(np.where(ys[num_testing:]==0)[0])
        nonzero_pos = list(np.where(ys[num_testing:]!=0)[0])
        for p in zero_pos:
            p += num_testing
            p_feature = features[p]
            p_adjacencies = [sp.coo_matrix(adjacencies[p])]
            p_y = np.array(ys[p])
            for i in range((p%2)+1):
                q = random.choice(nonzero_pos) + num_testing
                p_feature = np.append(p_feature, features[q], axis=0)
                p_adjacencies.append(sp.coo_matrix(adjacencies[q]))
                p_y += ys[q]
            if p % 5 == 0:
                q = np.random.randint(num_testing, data_n)
                p_feature = np.append(p_feature, features[q], axis=0)
                p_adjacencies.append(sp.coo_matrix(adjacencies[q]))
                p_y += ys[q]
            if label_size == None or p_y[0] <= label_size[N][partition_part]:
                features.append(p_feature)
                adjacencies.append(sp.csr_array(sp.block_diag(p_adjacencies)))
                ys = np.append(ys, p_y.reshape(-1,1), axis=0)
    
    rows = [sp.coo_matrix(a).row for a in adjacencies]
    cols = [sp.coo_matrix(a).col for a in adjacencies]
    root_nodes = [get_root_node(col) for col in cols]

    features_test = features[:num_testing]
    rows_test = [sp.coo_matrix(a).row for a in adjacencies[:num_testing]]
    cols_test = [sp.coo_matrix(a).col for a in adjacencies[:num_testing]]
    ys_test = ys[:num_testing]
    root_nodes_test = root_nodes[:num_testing]

    features_train = features[num_testing:]
    rows_train = [sp.coo_matrix(a).row for a in adjacencies[num_testing:]]
    cols_train = [sp.coo_matrix(a).col for a in adjacencies[num_testing:]]
    ys_train = ys[num_testing:]
    root_nodes_train = root_nodes[num_testing:]

    return (
      InputData(features=features, rows=rows, columns=cols, labels=ys, root_nodes=root_nodes),
      InputData(features=features_train, rows=rows_train, columns=cols_train, labels=ys_train, root_nodes=root_nodes_train),
      InputData(features=features_test, rows=rows_test, columns=cols_test, labels=ys_test, root_nodes=root_nodes_test))




def batch(features, rows, cols, ys, root_nodes):
    """Converts a list of training examples into a batched single graph."""
    batch_size = len(features)
    max_features = max(f.shape[0] for f in features)
    b_features = np.zeros((batch_size, max_features, features[0].shape[1]))
    b_rows = []
    b_cols = []
    b_ys = np.zeros((batch_size, 1))
    b_masks = np.zeros((batch_size, max_features, 1))
    for i in range(batch_size):
        b_features[i, :features[i].shape[0], :] = features[i]
        b_rows.append(rows[i] + i * max_features)
        b_cols.append(cols[i] + i * max_features)
        b_ys[i, 0] = ys[i, 0]
        root_node = root_nodes[i]
        b_masks[i, root_node, 0] = 1.0

    b_features = b_features.reshape((-1, b_features.shape[-1]))
    b_rows = np.concatenate(b_rows)
    b_cols = np.concatenate(b_cols)

    return b_features, b_rows, b_cols, b_ys, b_masks


class Direction(enum.Enum):
    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    BOTH = enum.auto()


class Reduction(enum.Enum):
    SUM = enum.auto()
    MAX = enum.auto()


class MPNN(hk.Module):
    """Sparse Message-Passing Neural Network (Gilmer et al., ICML 2017)."""

    def __init__(
            self,
            *,
            out_size: int,
            mid_size: int,
            activation,
            direction: Direction,
            residual: bool,
            reduction: Reduction,
            message_relu: bool,
            with_bias: bool,
    ):
        """Build MPNN layer.

        Args:
          out_size: Output width of the network.
          mid_size: Width of the hidden layer.
          activation: Activation function to use before the layer output.
          direction: Direction of message passing. See Direction Enum.
          residual: Whether to use resiudal connections.
          reduction: Reduction function to aggregate messages at nodes. See
            Reduction enum.
          message_relu: Whether to apply a relu on each message.
          with_bias: Whether to add biases in linear layers.

        Returns:
          The output of the MPNN layer.
        """
        super().__init__(name='mpnn_aggr')
        self.mid_size = out_size if mid_size is None else mid_size
        self.out_size = out_size
        self.activation = activation
        self.direction = direction
        self.reduction = reduction
        self.residual = residual
        self.message_relu = message_relu
        self.with_bias = with_bias

        @jax.jit
        def jax_coo_sum(rows, cols, msg_in, msg_out):
            msg_vect = msg_in[rows] + msg_out[cols]
            if message_relu:
                msg_vect = jax.nn.relu(msg_vect)
            return jnp.zeros_like(msg_out).at[rows].add(msg_vect)

        @jax.jit
        def jax_coo_max(rows, cols, msg_in, msg_out):
            msg_vect = msg_in[rows] + msg_out[cols]
            if message_relu:
                msg_vect = jax.nn.relu(msg_vect)
            return jnp.zeros_like(msg_in).at[rows].max(msg_vect)

        self.jax_coo_sum = jax_coo_sum
        self.jax_coo_max = jax_coo_max

    def __call__(self, features, rows, cols):
        if self.direction == Direction.FORWARD or self.direction == Direction.BOTH:
            m1_1 = hk.Linear(self.mid_size, with_bias=self.with_bias)
            m2_1 = hk.Linear(self.mid_size, with_bias=self.with_bias)
            msg_1_1 = m1_1(features)
            msg_2_1 = m2_1(features)
        if self.direction == Direction.BACKWARD or self.direction == Direction.BOTH:
            m1_2 = hk.Linear(self.mid_size, with_bias=self.with_bias)
            m2_2 = hk.Linear(self.mid_size, with_bias=self.with_bias)
            msg_1_2 = m1_2(features)
            msg_2_2 = m2_2(features)

        o2 = hk.Linear(self.out_size, with_bias=self.with_bias)

        if self.reduction == Reduction.MAX:
            reduction = self.jax_coo_max
        elif self.reduction == Reduction.SUM:
            reduction = self.jax_coo_sum
        else:
            raise ValueError('Unknown reduction %s' % self.reduction)

        if self.direction == Direction.FORWARD:
            msgs = reduction(rows, cols, msg_1_1, msg_2_1)
        elif self.direction == Direction.BACKWARD:
            msgs = reduction(cols, rows, msg_1_2, msg_2_2)
        elif self.direction == Direction.BOTH:
            msgs_1 = reduction(rows, cols, msg_1_1, msg_2_1)
            msgs_2 = reduction(cols, rows, msg_1_2, msg_2_2)
            msgs = jnp.concatenate([msgs_1, msgs_2], axis=-1)
            pass
        else:
            raise ValueError('Unknown direction %s' % self.direction)

        h_2 = o2(msgs)
        if self.residual:
            o1 = hk.Linear(self.out_size, with_bias=self.with_bias)
            h_1 = o1(features)
            network_output = h_1 + h_2
        else:
            network_output = h_2

        if self.activation is not None:
            network_output = self.activation(network_output)

        return network_output


class Model:

    def __init__(
            self,
            *,
            num_layers: int,
            num_features: int,
            num_classes: int,
            direction: Direction,
            reduction: Reduction,
            apply_relu_activation: bool,
            use_mask: bool,
            share: bool,
            message_relu: bool,
            with_bias: bool,
    ):
        """Get the jax model function and associated functions.

        Args:
          num_layers: The number of layers in the GraphNet - equivalently the number
            of propagation steps.
          num_features: The dimension of the hidden layers / messages.
          num_classes: The number of target classes.
          direction: Edges to pass messages along, see Direction enum.
          reduction: The reduction operation to be used to aggregate messages at
            each node at each step. See Reduction enum.
          apply_relu_activation: Whether to apply a relu at the end of each
            propogration step.
          use_mask: Boolean; should a masked prediction in central node be
            performed?
          share: Boolean; should the GNN layers be shared?
          message_relu: Boolean; should a ReLU be used in the message function?
          with_bias: Boolean; should the linear layers have bias?
        """
        self._num_layers = num_layers
        self._num_features = num_features
        self._num_classes = num_classes
        self._direction = direction
        self._reduction = reduction
        self._apply_relu_activation = apply_relu_activation
        self._use_mask = use_mask
        self._share = share
        self._message_relu = message_relu
        self._with_bias = with_bias

    def _kl_net(self, features, rows, cols, batch_size, masks):
        in_enc = hk.Linear(self._num_features)

        if self._apply_relu_activation:
            activation_fn = jax.nn.relu
        else:
            activation_fn = lambda net: net

        gnns = []
        for i in range(self._num_layers):
            if i == 0 or not self._share:
                gnns.append(
                    MPNN(
                        out_size=self._num_features,
                        mid_size=None,
                        direction=self._direction,
                        reduction=self._reduction,
                        activation=activation_fn,
                        message_relu=self._message_relu,
                        with_bias=self._with_bias,
                        residual=True))
            else:
                gnns.append(gnns[-1])

        out_enc = hk.Linear(self._num_classes, with_bias=self._with_bias)

        hiddens = []
        hidden = in_enc(features)
        hiddens.append(jnp.reshape(hidden, (batch_size, -1, self._num_features)))
        for gnn in gnns:
            hidden = gnn(hidden, rows, cols)
            hiddens.append(jnp.reshape(hidden, (batch_size, -1, self._num_features)))

        hidden = jnp.reshape(hidden, (batch_size, -1, self._num_features))

        if self._use_mask:
            h_bar = jnp.sum(hidden * masks, axis=1)
        else:
            h_bar = jnp.max(hidden, axis=1)

        lgts = out_enc(h_bar)

        return hiddens, lgts

    @property
    def net(self):
        return hk.transform(self._kl_net)

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, params, features, rows, cols, ys, masks):
        _, lgts = self.net.apply(params, None, features, rows, cols, ys.shape[0],
                                    masks)
        return -jnp.mean(
            jax.nn.log_softmax(lgts) *
            jnp.squeeze(jax.nn.one_hot(ys, self._num_classes), 1))

    @functools.partial(jax.jit, static_argnums=(0,))
    def accuracy(self, params, features, rows, cols, ys, masks):
        _, lgts = self.net.apply(params, None, features, rows, cols, ys.shape[0],
                                    masks)
        pred = jnp.argmax(lgts, axis=-1)
        true_vals = jnp.squeeze(ys, axis=1)
        acc = jnp.mean(pred == true_vals)
        return acc

def get_baseline_accuracy(labels):
    return max(np.bincount(labels[:, 0])) / labels.shape[0]

def get_salience_vectors(salience_fn, params, full_dataset, batch_size=100):
    salient_features_arr = []
    for i in range(0, len(full_dataset.features), batch_size):
        b_features, b_rows, b_cols, b_ys, b_masks = batch(
            full_dataset.features[i:i + batch_size],
            full_dataset.rows[i:i + batch_size],
            full_dataset.columns[i:i + batch_size],
            full_dataset.labels[i:i + batch_size],
            full_dataset.root_nodes[i:i + batch_size],
        )
        salient_features = salience_fn(params, b_features, b_rows, b_cols, b_ys, 
                                        b_masks)
        effective_batch_size = len(full_dataset.features[i:i + batch_size])
        salient_features_arr.extend(
            np.reshape(salient_features, [effective_batch_size, -1, full_dataset.features[0].shape[1]]))
    return salient_features_arr
