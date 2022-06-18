import os

N = 7
partition_part = 1
num_layers = 5
num_features = 64
num_epochs = 100
batch_size = 200
use_pretrained_weights = True
save_trained_weights = True

label_size = {7: [0, 60, 36, 35, 28, 38, 58, 85]}

step_size = 0.001
train_fraction = .8

DIR_PATH = '/root/Hwang/mathematics_conjectures/Hwang'
GRAPH_DIR = os.path.join(DIR_PATH, f'Data/N_{N}')
PARAM_DIR = os.path.join(DIR_PATH, 'Parameters')
PARAM_FILE = os.path.join(PARAM_DIR, f'parameters_{N}_{partition_part}_{num_layers}_{num_features}')
for key in feature_list.keys():
    PARAM_FILE += f'_{key}'
PARAM_FILE += '.pickle'
