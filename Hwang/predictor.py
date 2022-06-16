from calendar import EPOCH
from utils import *

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

trained_params = [None]
models = [None]
for partition_part in range(1, N+1):
    PARAM_FILE = os.path.join(PARAM_DIR, f'parameters_{N}_{partition_part}_{num_layers}_{num_features}')
    for key in feature_list.keys():
        PARAM_FILE += f'_{key}'
    PARAM_FILE += '.pickle'
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

def is_compatible(P, a, b):
    if P[a-1] < b or P[b-1] < a:
        return True
    return False

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
    row = sp.coo_matrix(adj).row
    col = sp.coo_matrix(adj).col
    return feature, row, col

def predictor(P, word, params, models):
    feat, row, col = get_graph_datum(P, word)
    partition = []
    for i in range(len(P), 0, -1):
        _, lgts = models[i].net.apply(params[i], None, feat, row, col, 1, None)
        mult = jnp.argmax(jax.nn.log_softmax(lgts)[0])
        for j in range(mult):
            partition.append(i)
    return partition