# import warnings
#
# warnings.filterwarnings('ignore')


import numpy as np
from data_loader import batch



def get_baseline_accuracy(labels):
    return max(np.bincount(labels[:, 0])) / labels.shape[0]


def get_salience_vectors(salience_fn, params, full_dataset, batch_size=100):
    salient_features_arr = []
    for i in range(0, len(full_dataset.features), batch_size):
        b_features, b_rows_1, b_cols_1, b_rows_2, b_cols_2, b_ys, b_masks = batch(
            full_dataset.features[i:i + batch_size],
            full_dataset.rows_1[i:i + batch_size],
            full_dataset.columns_1[i:i + batch_size],
            full_dataset.rows_2[i:i + batch_size],
            full_dataset.columns_2[i:i + batch_size],
            full_dataset.labels[i:i + batch_size],
            full_dataset.root_nodes[i:i + batch_size],
        )
        salient_features = salience_fn(params, b_features, b_rows_1, b_cols_1, b_rows_2, b_cols_2, b_ys,
                                       b_masks)
        effective_batch_size = len(full_dataset.features[i:i + batch_size])
        salient_features_arr.extend(
            np.reshape(salient_features, [effective_batch_size, -1, full_dataset.features[0].shape[1]]))
    return salient_features_arr


def compute_accuracies(model,params_to_evaluate, dataset, batch_size=100):
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


def print_accuracies(model, params_to_evaluate, dataset_test, dataset_train, batch_size=100):
    train_accuracy = compute_accuracies(
        model, params_to_evaluate, dataset=dataset_train, batch_size=batch_size)
    test_accuracy = compute_accuracies(
        model, params_to_evaluate, dataset=dataset_test, batch_size=batch_size)

    combined_accuracy = np.average(
        [train_accuracy, test_accuracy],
        weights=[len(dataset_train.features),
                 len(dataset_test.features)])
    print(f'Train accuracy: {train_accuracy:.3f} | '
          f'Test accuracy: {test_accuracy:.3f} | '
          f'Combined accuracy: {combined_accuracy:.3f}')
    return train_accuracy, test_accuracy, combined_accuracy

def print_test_accuracies(model, params_to_evaluate, dataset_test,batch_size=100):
    test_accuracy = compute_accuracies(
        model, params_to_evaluate, dataset=dataset_test, batch_size=batch_size)
    print(f'Test accuracy: {test_accuracy:.3f} | ')
    return  test_accuracy

