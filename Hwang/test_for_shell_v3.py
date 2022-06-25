from calendar import EPOCH
from utils_v3 import *

import warnings
warnings.filterwarnings('ignore')

def compute_accuracies(params_to_evaluate, dataset, batch_size=100):
    total_correct = 0.0
    for i in range(0, len(dataset.features), batch_size):
        b_features, b_rows_1, b_cols_1, b_rows_2, b_cols_2, b_ys, b_masks = batch(
            dataset.features[i:i + batch_size],
            dataset.rows_1[i:i + batch_size],
            dataset.columns_1[i:i + batch_size],
            dataset.rows_2[i:i + batch_size],
            dataset.columns_2[i:i + batch_size],
            dataset.labels[i:i + batch_size],
            dataset.root_nodes[i:i + batch_size],
        )

        accs = model.accuracy(params_to_evaluate, b_features, b_rows_1, b_cols_1, b_rows_2, b_cols_2, b_ys,
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


def train(params, opt_state, features, rows_1, cols_1, rows_2, cols_2, ys, masks):
    curr_loss, gradient = loss_val_gr(params, features, rows_1, cols_1, rows_2, cols_2, ys, masks)
    updates, opt_state = opt_update(gradient, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, curr_loss

print("Loading input data...")
full_dataset, train_dataset, test_dataset = load_input_data(N, partition_part, feature_list,
                                                            extended=True,
                                                            label_size=label_size)

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
            rows_1=train_dataset.rows_1[0],
            cols_1=train_dataset.columns_1[0],
            rows_2=train_dataset.rows_2[0],
            cols_2=train_dataset.columns_2[0],
            batch_size=1,
            masks=train_dataset.features[0][np.newaxis, :, :])
    
    trained_opt_state = opt_init(trained_params)
    for ep in range(1, num_epochs + 1):
        tr_data = list(
            zip(
                train_dataset.features,
                train_dataset.rows_1,
                train_dataset.columns_1,
                train_dataset.rows_2,
                train_dataset.columns_2,
                train_dataset.labels,
                train_dataset.root_nodes,
            ))
        random.shuffle(tr_data)
        features_train, rows_1_train, cols_1_train, rows_2_train, cols_2_train, ys_train, root_nodes_train = zip(
            *tr_data)

        features_train = list(features_train)
        rows_1_train = list(rows_1_train)
        cols_1_train = list(cols_1_train)
        rows_2_train = list(rows_2_train)
        cols_2_train = list(cols_2_train)
        ys_train = np.array(ys_train)
        root_nodes_train = list(root_nodes_train)

        for i in range(0, len(features_train), batch_size):
            b_features, b_rows_1, b_cols_1, b_rows_2, b_cols_2, b_ys, b_masks = batch(
                features_train[i:i + batch_size],
                rows_1_train[i:i + batch_size],
                cols_1_train[i:i + batch_size],
                rows_2_train[i:i + batch_size],
                cols_2_train[i:i + batch_size],
                ys_train[i:i + batch_size],
                root_nodes_train[i:i + batch_size],
            )

            trained_params, trained_opt_state, curr_loss = train(
                trained_params,
                trained_opt_state,
                b_features,
                b_rows_1,
                b_cols_1,
                b_rows_2,
                b_cols_2,
                b_ys,
                b_masks,
            )

            accs = model.accuracy(
                trained_params,
                b_features,
                b_rows_1,
                b_cols_1,
                b_rows_2,
                b_cols_2,
                b_ys,
                b_masks,
            )
        print(datetime.datetime.now(), f"Epoch {ep:2d} completed!")
except Exception as ex:
    print(f"The following exception occurs: {ex}")

if save_trained_weights:
    with open(PARAM_FILE, 'wb') as f:
        pickle.dump(trained_params, f)

# with open("logs.out", "a") as f:
#     f.write("========================================================================\n")
#     f.write(f'{datetime.datetime.now()} Training completed!\n')
#     f.write(f"N, Partition part: {N}, {partition_part}\n")
#     f.write(f"Number of layers: {num_layers}\n")
#     f.write(f"Number of epochs: {ep}\n")
#     f.write(f"Batch size: {batch_size}\n")
#     f.write("List of features:\n")
#     for feature in feature_list.keys():
#         f.write(f"\t{feature}\n")
    
#     train_accuracy, test_accuracy, combined_accuracy = print_accuracies(trained_params, test_dataset, train_dataset, batch_size)
#     f.write("\n--Result--\n")
#     f.write("                  Train |  Test | Combined\n")
#     f.write(f'Baseline accuracy {get_baseline_accuracy(train_dataset.labels):.3f} | '
#             f'{get_baseline_accuracy(test_dataset.labels):.3f} | '
#             f'{get_baseline_accuracy(full_dataset.labels):.3f}\n')
#     f.write(f'Model accuracy    {train_accuracy:.3f} | '
#             f'{test_accuracy:.3f} | '
#             f'{combined_accuracy:.3f}\n')
