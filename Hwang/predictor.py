from generate_data import *
from utils_v3 import *

import warnings
warnings.filterwarnings('ignore')


#################################
####### Model information #######
#################################

# N = 7
# partition_part = 6
# num_layers = 5
# num_features = 64

# feature_list = {
# #             'in_centrality': nx.in_degree_centrality,
# #             'out_centrality': nx.out_degree_centrality,
# #             'shortest_path_length': shortest_path_lengths,
# #             'longest_path_length': longest_path_lengths,
# #             'random_feature': random_feature,
#             'constant_feature': constant_feature,
# #             'numbering_feature': numbering_feature,
#         }

# label_size = {7: [0, 60, 36, 35, 28, 38, 58, 85]}

with open("PartitionIndex.json", "r") as f:
    PartitionIndex = json.load(f)
with open("TransitionMatrix.json", "r") as f:
    TMs = json.load(f)
with open("Partitions.json", "r") as f:
    Partitions = json.load(f)
with open("PartitionMultiplicity.json", "r") as f:
    PartitionMultiplicity = json.load(f)

trained_params = [None]
models = [None]
for partition_part in range(1, N+1):
    PARAM_FILE = os.path.join(PARAM_DIR, f'parameters_{N}_{partition_part}_{num_layers}_{num_features}')
    for key in feature_list.keys():
        PARAM_FILE += f'_{key}'
    PARAM_FILE += '_v3.pickle'
    with open(PARAM_FILE, 'rb') as f:
        trained_params.append(pickle.load(f))
    models.append(Model(num_layers=num_layers,
                       num_features=num_features,
                       num_classes=label_size[N][partition_part],
                       direction=Direction.BOTH,
                       reduction=Reduction.SUM,
                       apply_relu_activation=True,
                       use_mask=False,
                       share=False,
                       message_relu=True,
                       with_bias=True))

def make_graph(P, word):
    n = len(P)
    row = []
    col = []
    data = []
    for i in range(1,n):
        for j in range(i):
            if is_compatible(P, word[i], word[j]) == False and is_compatible(P, word[j], word[i]) == False:
                row.append(word[i]-1)
                col.append(word[j]-1)
                data.append(1)
    return sp.coo_matrix((data, (row,col)), shape=(n,n)), sp.coo_matrix((data, (row,col)), shape=(n,n)) + sp.eye(n)

def get_graph_datum(P, word):
    D, adj = make_graph(P, word)
    feature = get_feature(nx.from_scipy_sparse_matrix(D))
    row = sp.coo_matrix(adj).row
    col = sp.coo_matrix(adj).col
    row_1 = []
    col_1 = []
    row_2 = []
    col_2 = []
    for i in range(len(row)):
        if row[i] >= col[i]:
            row_1.append(row[i])
            col_1.append(col[i])
        if row[i] <= col[i]:
            row_2.append(row[i])
            col_2.append(col[i])
    row_1[i] = np.array(row_1[i], dtype=np.int8)
    col_1[i] = np.array(col_1[i], dtype=np.int8)
    row_2[i] = np.array(row_2[i], dtype=np.int8)
    col_2[i] = np.array(col_2[i], dtype=np.int8)
    return feature, row_1, col_1, row_2, col_2

def get_feature(graph):
    feat_dict = dict()
    for key, feat in feature_list.items():
        feat_dict[key] = feat(graph)
    curr_feature = np.zeros((len(graph), len(feat_dict)))
    for n, node in enumerate(graph.nodes):
        for i, (name, value) in enumerate(feat_dict.items()):
            curr_feature[n, i] = value[node]
    return curr_feature

def get_all_equiv_words(P, word):
    queue = [word]
    for word in queue:
        word_list = get_equiv_words(P, word)
        for eqv_word in word_list:
            if not eqv_word in queue:
                queue.append(eqv_word)
    return queue

def predictor(P, word):
    feat, row_1, col_1, row_2, col_2 = get_graph_datum(P, word)
    partition = []
    for i in range(len(P), 0, -1):
        _, lgts = models[i].net.apply(trained_params[i], None, feat, row_1, col_1, row_2, col_2, 1, None)
        mult = jnp.argmax(jax.nn.log_softmax(lgts)[0])
        for j in range(mult):
            partition.append(i)
    return partition

def predictor_orbit(P, word):
    equiv_words = get_all_equiv_words(P, word)
    noDes = get_noDesWords(P, equiv_words)
    result = []
    for word in noDes:
        result.append(predictor(P, word))
    return result

def answer(P, word):
    n = len(P)
    equiv_words = get_all_equiv_words(P, word)
    pars = K_H(P, equiv_words)
    result = []
    for i in range(len(pars)):
        for j in range(int(pars[i])):
            result.append(Partitions[str(n)][i])
    return result, pars