from generate_data import *
from utils_modified import *

import warnings
warnings.filterwarnings('ignore')


#################################
####### Model information #######
#################################

N = 7
partition_part = 6
num_layers = 5
num_features = 64

feature_list = {
#             'in_centrality': nx.in_degree_centrality,
#             'out_centrality': nx.out_degree_centrality,
#             'shortest_path_length': shortest_path_lengths,
#             'longest_path_length': longest_path_lengths,
#             'random_feature': random_feature,
            'constant_feature': constant_feature,
#             'numbering_feature': numbering_feature,
        }

label_size = {7: [0, 60, 36, 35, 28, 38, 58, 85]}

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
    PARAM_FILE += '_v2.pickle'
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
            if is_compatible(P, word[i], word[j]) == False:
                row.append(word[i]-1)
                col.append(word[j]-1)
                data.append(1)
    return sp.coo_matrix((data, (row,col)), shape=(n,n)) + sp.eye(n)

def get_graph_datum(P, word):
    adj = make_graph(P, word)
    feature = np.ones((N, 1))
    row_1 = sp.coo_matrix(adj).row
    col_1 = sp.coo_matrix(adj).col
    row_2 = sp.coo_matrix(adj).row
    col_2 = sp.coo_matrix(adj).col
    for i in range(len(row_2)):
        if row_2[i] > col_2[i]:
            temp = row_2[i]
            row_2[i] = col_2[i]
            col_2[i] = temp
    return feature, row_1, col_1, row_2, col_2

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
    equiv_words = get_all_equiv_words(P, word)
    pars = K_H(P, equiv_words)
    result = []
    for i in range(len(pars)):
        for j in range(int(pars[i])):
            result.append(Partitions[str(N)][i])
    return result, pars