from generate_data import *
from utils import *
from data_loader import *
from Model import *
from training_info import *
import pickle

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

with open(os.path.join(JSON_DIR, 'PartitionIndex.json'), "r") as f:
    PartitionIndex = json.load(f)
with open(os.path.join(JSON_DIR, 'TransitionMatrix.json'), "r") as f:
    TMs = json.load(f)
with open(os.path.join(JSON_DIR, 'Partitions.json'), "r") as f:
    Partitions = json.load(f)
with open(os.path.join(JSON_DIR, 'PartitionMultiplicity.json'), "r") as f:
    PartitionMultiplicity = json.load(f)

# trained_params = [None]
# models = [None]
# for partition_part in range(1, N+1):
#     PARAM_FILE = os.path.join(PARAM_DIR, f'parameters_{N}_{partition_part}_{num_layers}_{num_features}')
#     for key in feature_list.keys():
#         PARAM_FILE += f'_{key}'
#     PARAM_FILE += '.pickle'
#     with open(PARAM_FILE, 'rb') as f:
#         trained_params.append(pickle.load(f))
#     models.append(Model_list(num_layers=num_layers,
#                        num_features=num_features,
#                        num_classes=label_size[N][partition_part],
#                        direction=Direction.BOTH,
#                        reduction=Reduction.SUM,
#                        apply_relu_activation=True,
#                        use_mask=False,
#                        share=False,
#                        message_relu=True,
#                        with_bias=True))


PARAM_FILE = os.path.join(PARAM_DIR, f'parameters_{N}_{partition_parts}_{num_layers}_{num_features}')
for key in feature_list.keys():
    PARAM_FILE += f'_{key}'
PARAM_FILE += '.pickle'
with open(PARAM_FILE, 'rb') as f:
    trained_param = pickle.load(f)
model = Model_list(num_layers=num_layers,
                    num_features=num_features,
                    num_classes=np.max(np.array(label_size[N])[partition_parts]),
                    size_graph=N,
                    direction=Direction.BOTH,
                    reduction=Reduction.SUM,
                    apply_relu_activation=True,
                    use_mask=False,
                    share=False,
                    message_relu=True,
                    with_bias=True)

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
    feature = get_feature(nx.from_scipy_sparse_matrix(D, create_using=nx.DiGraph))
    row = sp.coo_matrix(adj).row
    col = sp.coo_matrix(adj).col
    
    row_1 = np.array(row, dtype=np.int16)
    col_1 = np.array(col, dtype=np.int16)
    Hasse_row, Hasse_col = Hasse_diagram(row, col)
    row_2, col_2 = go_right(Hasse_row, Hasse_col)

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
    print(PARAM_FILE)
    _, lgts = model.net.apply(trained_param, None, feat, row_1, col_1, row_2, col_2, 1, None)
    print(lgts.shape)
#     print(lgts)
    print(jnp.argmax(lgts, axis=-1))
    partition = []
    for i in range(0, len(P)):
        mult = jnp.argmax(lgts[0][i])
        partition.append(mult)
    return partition
    for i in range(len(P), 0, -1):
        mult = jnp.argmax(lgts[0][i])
        for j in range(mult):
            partition.append(i)
    return partition

    # partition = []
    # for i in range(len(P), 0, -1):
    #     _, lgts = models[i].net.apply(trained_params[i], None, feat, row_1, col_1, row_2, col_2, 1, None)
    #     mult = jnp.argmax(jax.nn.log_softmax(lgts)[0])
    #     for j in range(mult):
    #         partition.append(i)
    # return partition

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