#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from calendar import EPOCH
import datetime
import pickle
import random
import optax
import os
# import psutil, sys, gc


# In[2]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# import jax
from training_info import *

from utils import print_test_accuracies
from Model import Model, Direction, Reduction, jax,Model_2, Model_list
from data_loader import load_input_data, batch
import jax.numpy as jnp

################################
# def train(params, opt_state, features, rows_1, cols_1, rows_2, cols_2, ys, masks):
#     curr_loss, gradient = loss_val_gr(params, features, rows_1, cols_1, rows_2, cols_2, ys, masks)
#     updates, opt_state = opt_update(gradient, opt_state)
#     new_params = optax.apply_updates(params, updates)
#     return new_params, opt_state, curr_loss

print("Loading input data...")
full_dataset, train_dataset, test_dataset = load_input_data(train_fraction, GRAPH_DIR, NUM_GRAPHS,
                                                            N, partition_parts, feature_list,
                                                            extended=True,
                                                            label_size=label_size)


num_classes = np.max(np.array(label_size[N])[partition_parts])
# model = Model(
#     num_layers=num_layers,
#     num_features=num_features,
#     num_classes=num_classes,
#     direction=Direction.BOTH,
#     reduction=Reduction.SUM,
#     apply_relu_activation=True,
#     use_mask=True, #False Originally
#     share=False,
#     message_relu=True,
#     with_bias=True)
model = Model_list(
    num_layers=num_layers,
    num_features=num_features,
    num_classes=num_classes,
    size_graph=N,
    direction=Direction.BOTH,
    reduction=Reduction.SUM,
    apply_relu_activation=True,
    use_mask=False,
    share=False,
    message_relu=True,
    with_bias=True)
loss_val_gr = jax.value_and_grad(model.loss)
opt_init, opt_update = optax.adam(step_size)


# In[3]:


# @title Perform training / Load pretrained weights
if use_pretrained_weights:
    try:
        print("Using pretrained weights")
        with open(PARAM_FILE, 'rb') as f:
            trained_params = pickle.load(f)
    except:
        print("There is no trained parameters")
        use_pretrained_weights = False
if use_pretrained_weights is False:
    print("Not using pretrained weights")
    trained_params = model.net.init(
        jax.random.PRNGKey(42),
        features=train_dataset.features[0],
        rows_1=train_dataset.rows_1[0],
        cols_1=train_dataset.columns_1[0],
        rows_2=train_dataset.rows_2[0],
        cols_2=train_dataset.columns_2[0],
        batch_size=1,
        masks=train_dataset.features[0][np.newaxis, :, :])

trained_opt_state = opt_init(trained_params)#22237 initial memory use


# In[4]:


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
    # root_nodes_train = jnp.ones_like(np.array(root_nodes_train))
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
        b_masks = jnp.ones_like(b_masks) / N
        curr_loss, gradient = loss_val_gr(trained_params, b_features, b_rows_1, b_cols_1,
                                          b_rows_2, b_cols_2, b_ys, b_masks)
        updates, trained_opt_state = opt_update(gradient, trained_opt_state)
        trained_params = optax.apply_updates(trained_params, updates)

        accs = model.accuracy(trained_params, b_features, b_rows_1, b_cols_1,
                              b_rows_2, b_cols_2, b_ys, b_masks, )#22512 ? 22427?22475?22523
        print(datetime.datetime.now(),
              f"Iteration {i:5d} | Batch loss {curr_loss:.6f}",
              f"Batch accuracy {accs:.2f}")
    print(datetime.datetime.now(), f"Epoch {ep:2d} completed!")
    print(datetime.datetime.now(), f"Epoch {ep:2d}       | ", end="")
    print_test_accuracies(model, trained_params,test_dataset, batch_size)


# In[ ]:


ys=ys_train[i:i + batch_size]


# In[ ]:


ys.shape


# In[ ]:


jnp.argmax(ys,axis=-1).shape


# In[ ]:


ohys=jax.nn.one_hot(ys, num_classes)


# In[ ]:


ohys.shape


# In[ ]:


jnp.squeeze(ys,axis=1)


# In[ ]:


jnp.mean(jnp.argmax(ohys,axis=-1)==ys)

