from calendar import EPOCH
from utils import *

import warnings
warnings.filterwarnings('ignore')


################################
##### Training information #####
################################

N = 7
partition_part = 1
num_layers = 5
num_features = 64
num_epochs = 100
batch_size = 200
use_pretrained_weights = True
save_trained_weights = True

step_size = 0.001

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

PARAM_FILE = os.path.join(PARAM_DIR, f'parameters_{N}_{partition_part}_{num_layers}_{num_features}')
for key in feature_list.keys():
    PARAM_FILE += f'_{key}'
PARAM_FILE += '.pickle'

################################


def compute_accuracies(params_to_evaluate, dataset, batch_size=100):
    total_correct = 0.0
    for i in range(0, len(dataset.features), batch_size):
        b_features, b_rows, b_cols, b_ys, b_masks = batch(
            dataset.features[i:i + batch_size], dataset.rows[i:i + batch_size],
            dataset.columns[i:i + batch_size], dataset.labels[i:i + batch_size],
            dataset.root_nodes[i:i + batch_size])

        accs = model.accuracy(params_to_evaluate, b_features, b_rows, b_cols, b_ys,
                              b_masks)
        total_correct += accs * len(dataset.features[i:i + batch_size])
    return total_correct / len(dataset.features)


def print_accuracies(params_to_evaluate,
                     dataset_test,
                     dataset_train,
                     batch_size=100):
    train_accuracy = compute_accuracies(
        params_to_evaluate, dataset=train_dataset, batch_size=batch_size)
    test_accuracy = compute_accuracies(
        params_to_evaluate, dataset=test_dataset, batch_size=batch_size)

    combined_accuracy = np.average(
        [train_accuracy, test_accuracy],
        weights=[len(dataset_train.features),
                 len(dataset_test.features)])
    print(f'Train accuracy: {train_accuracy:.3f} | '
          f'Test accuracy: {test_accuracy:.3f} | '
          f'Combined accuracy: {combined_accuracy:.3f}')
    return train_accuracy, test_accuracy, combined_accuracy


def train(params, opt_state, features, rows, cols, ys, masks):
    curr_loss, gradient = loss_val_gr(params, features, rows, cols, ys, masks)
    updates, opt_state = opt_update(gradient, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, curr_loss

print("Loading input data...")
full_dataset, train_dataset, test_dataset = load_input_data(N, partition_part, feature_list, label_size)

# @title Network Setup

num_classes = label_size[N][partition_part]
model = Model(
    num_layers=num_layers,
    num_features=num_features,
    num_classes=num_classes,
    direction=Direction.BOTH,
    reduction=Reduction.SUM,
    apply_relu_activation=True,
    use_mask=False,
    share=False,
    message_relu=True,
    with_bias=True)

loss_val_gr = jax.value_and_grad(model.loss)
opt_init, opt_update = optax.adam(step_size)

# In[28]:


# @title Perform training / Load pretrained weights
try:
    if use_pretrained_weights:
        try:
            with open(PARAM_FILE, 'rb') as f:
                trained_params = pickle.load(f)
        except:
            print("There is no trained parameters")
            use_pretrained_weights = False
    if use_pretrained_weights == False:
        trained_params = model.net.init(
            jax.random.PRNGKey(42),
            features=train_dataset.features[0],
            rows=train_dataset.rows[0],
            cols=train_dataset.columns[0],
            batch_size=1,
            masks=train_dataset.features[0][np.newaxis, :, :])
    
    trained_opt_state = opt_init(trained_params)
    for ep in range(1, num_epochs + 1):
        tr_data = list(
            zip(
                train_dataset.features,
                train_dataset.rows,
                train_dataset.columns,
                train_dataset.labels,
                train_dataset.root_nodes,
            ))
        random.shuffle(tr_data)
        features_train, rows_train, cols_train, ys_train, root_nodes_train = zip(
            *tr_data)

        features_train = list(features_train)
        rows_train = list(rows_train)
        cols_train = list(cols_train)
        ys_train = np.array(ys_train)
        root_nodes_train = list(root_nodes_train)

        for i in range(0, len(features_train), batch_size):
            b_features, b_rows, b_cols, b_ys, b_masks = batch(
                features_train[i:i + batch_size],
                rows_train[i:i + batch_size],
                cols_train[i:i + batch_size],
                ys_train[i:i + batch_size],
                root_nodes_train[i:i + batch_size],
            )

            trained_params, trained_opt_state, curr_loss = train(
                trained_params,
                trained_opt_state,
                b_features,
                b_rows,
                b_cols,
                b_ys,
                b_masks,
            )

            accs = model.accuracy(
                trained_params,
                b_features,
                b_rows,
                b_cols,
                b_ys,
                b_masks,
            )
#             print(datetime.datetime.now(),
#                   f"Iteration {i:5d} | Batch loss {curr_loss:.6f}",
#                   f"Batch accuracy {accs:.2f}")

        print(datetime.datetime.now(), f"Epoch {ep:2d} completed!")

        # Calculate accuracy across full dataset once per epoch
        print(datetime.datetime.now(), f"Epoch {ep:2d}       | ", end="")
        print_accuracies(trained_params, test_dataset, train_dataset, batch_size)
except Exception as ex:
    print(f"The following exception occurs: {ex}")

print('Baseline accuracy', get_baseline_accuracy(train_dataset.labels))

# print('Computing saliences...')
# salience_fn = jax.jit(jax.grad(lambda *args: jnp.sum(model.loss(*args)), 1))
# salient_features_arr = get_salience_vectors(salience_fn, trained_params,
#                                             full_dataset, batch_size)
# saliencies = np.linalg.norm(
#     np.concatenate(salient_features_arr, axis=0), axis=1)

# print(f"max saliency = {max(saliencies)}")

# cutoff = np.percentile(saliencies, 99)
# print(f"cutoff 0.99 = {cutoff}")



################################
if save_trained_weights:
    with open(PARAM_FILE, 'wb') as f:
        pickle.dump(trained_params, f)

with open("logs.out", "a") as f:
    f.write("========================================================================\n")
    f.write(f'{datetime.datetime.now()} Training completed!\n')
    f.write(f"N, Partition part: {N}, {partition_part}\n")
    f.write(f"Number of layers: {num_layers}\n")
    f.write(f"Number of epochs: {ep}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write("List of features:\n")
    for feature in feature_list.keys():
        f.write(f"\t{feature}\n")
    
    train_accuracy, test_accuracy, combined_accuracy = print_accuracies(trained_params, test_dataset, train_dataset, batch_size)
    f.write("\n--Result--\n")
    f.write("                  Train |  Test | Combined\n")
    f.write(f'Baseline accuracy {get_baseline_accuracy(train_dataset.labels):.3f} | '
            f'{get_baseline_accuracy(test_dataset.labels):.3f} | '
            f'{get_baseline_accuracy(full_dataset.labels):.3f}\n')
    f.write(f'Model accuracy    {train_accuracy:.3f} | '
            f'{test_accuracy:.3f} | '
            f'{combined_accuracy:.3f}\n')

