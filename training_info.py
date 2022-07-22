import os
from feature_functions import *

N = 7
partition_parts = [i for i in range(1,N+1)]
num_layers = 5
num_features = 64
num_epochs = 100
batch_size = 512
use_pretrained_weights = True
save_trained_weights = True

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

step_size = 0.001
train_fraction = .8

DIR_PATH = '/root/Hwang/mathematics_conjectures'
GRAPH_DIR = os.path.join(DIR_PATH, f'Hwang/Data/N_{N}')
PARAM_DIR = os.path.join(DIR_PATH, 'Parameters')
NUM_GRAPHS = len([f for f in os.listdir(GRAPH_DIR) if f.startswith("graph_")])
PARAM_FILE = os.path.join(PARAM_DIR, f'parameters_{N}_{partition_parts}_{num_layers}_{num_features}')
for key in feature_list.keys():
    PARAM_FILE += f'_{key}'
PARAM_FILE += '.pickle'