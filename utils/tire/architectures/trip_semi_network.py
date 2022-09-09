import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

from .triplet_semi import *


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, windows, labels, batch_size=128, shuffle=True):
        super().__init__()
        self.windows = windows
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.key_array = np.arange(self.windows.shape[0], dtype=np.uint32)

        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = self.key_array[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        x = np.asarray(self.windows[keys], dtype=np.float32)
        y = np.asarray(self.labels[keys], dtype=np.float32)

        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)


def create_parallel_aes(
    window_size_per_ae, intermediate_dim=10, latent_dim=1, nr_shared=1, nr_ae=2
):
    """
    Create a Tensorflow model with parallel autoencoders, as visualized in Figure 1 of the TIRE paper.

    Args:
        window_size_per_ae: window size for the AE
        intermediate_dim: intermediate dimension for stacked AE, for single-layer AE use 0
        latent_dim: latent dimension of AE
        nr_ae: number of parallel AEs (K in paper)
        nr_shared: number of shared features (should be <= latent_dim)
        loss_weight: lambda in paper

    Returns:
        A parallel AE model instance, its encoder part and its decoder part
    """

    # # initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None)
    initializer = tf.keras.initializers.GlorotUniform()
    wspa = window_size_per_ae
    x = Input(
        shape=(
            nr_ae,
            wspa,
        ),
        name="data",
    )

    if intermediate_dim == 0:
        y = x
    else:
        y = Dense(
            intermediate_dim,
            kernel_initializer=initializer,
            activation=tf.nn.relu,
            name="enc1",
        )(x)
        y = tf.keras.layers.BatchNormalization()(y)

    # y = Dense(intermediate_dim, activation=tf.nn.relu, name='add1')(y)
    # y = tf.keras.layers.BatchNormalization()(y)
    z_shared = Dense(
        nr_shared,
        kernel_initializer=initializer,
        activation=tf.nn.tanh,
        name="enc2-1",
    )(y)
    z_unshared = Dense(
        latent_dim - nr_shared,
        kernel_initializer=initializer,
        activation=tf.nn.tanh,
        name="enc2-2",
    )(y)
    z = tf.concat([z_shared, z_unshared], -1)

    if intermediate_dim == 0:
        y = z
    else:
        y = Dense(
            intermediate_dim,
            kernel_initializer=initializer,
            activation=tf.nn.relu,
            name="dec1",
        )(z)
        y = tf.keras.layers.BatchNormalization()(y)

    # y = Dense(intermediate_dim, activation=tf.nn.relu, name='add2')(y)
    # y = tf.keras.layers.BatchNormalization()(y)
    x_decoded = Dense(
        wspa, kernel_initializer=initializer, activation=tf.nn.tanh, name="dec2"
    )(y)

    pae = Model(x, x_decoded)
    # pae.load_weights("weights.h5", by_name=True)

    encoder = Model(x, z)

    return pae, encoder


def prepare_input_paes(windows, nr_ae=2):
    new_windows = []
    nr_windows = windows.shape[0]
    for i in range(nr_ae):
        new_windows.append(windows[i : nr_windows - nr_ae + 1 + i])
    return np.transpose(new_windows, (1, 0, 2))


def train_AE(
    ori_windows,
    labels,
    intermediate_dim=0,
    latent_dim=1,
    nr_shared=1,
    loss_weight_triplet=1,
    batch_all=True,
    margin=0.1,
    squared=True,
    nr_epochs=200,
    batch_size=256,
    seed=None,
):
    """
    Creates and trains an autoencoder with a Time-Invariant REpresentation (TIRE)

    Args:
        windows: time series windows (i.e. {y_t}_t or {z_t}_t in the notation of the paper)
        intermediate_dim: intermediate dimension for stacked AE, for single-layer AE use 0
        latent_dim: latent dimension of AE
        nr_shared: number of shared features (should be <= latent_dim)
        nr_ae: number of parallel AEs (K in paper)
        loss_weight: lambda in paper
        nr_epochs: number of epochs for training
        nr_patience: patience for early stopping

    Returns:
        returns the TIRE encoded windows for all windows
    """
    if seed is not None:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def generate_data(windows, labels):
        adjency_slice = []
        adj_order_list = []
        # negative_slice = []
        # neg_order_list = []
        nr_windows = windows.shape[0]
        anchors = windows[0 : nr_windows - 1]
        anchor_slice = np.expand_dims(np.array(anchors), axis=1)
        for idx, label in enumerate(labels[:-1]):
            if label == labels[idx + 1]:
                adj_order_list.append(idx + 1)
            else:
                adj_order_list.append(idx - 1)
            # neg_candidates = np.concatenate((np.argwhere(labels == label + 1), np.argwhere(labels == label - 1)), axis=0)
            # index = np.random.choice(neg_candidates.flatten(), 1)[0]
            # neg_order_list.append(index)

        for idx in adj_order_list:
            adjency_slice.append(windows[idx, :])
        adjency_slice = np.expand_dims(np.array(adjency_slice), axis=1)

        # for idx in neg_order_list:
        #     negative_slice.append(windows[idx, :])
        # negative_slice = np.expand_dims(np.array(negative_slice), axis=1)
        return np.concatenate((anchor_slice, adjency_slice), axis=1)

    loss_train = np.zeros(shape=(nr_epochs,), dtype=np.float32)
    # windows = ori_windows
    windows = generate_data(ori_windows, labels)
    window_size_per_ae = windows.shape[-1]
    # encoded_labels = tf.keras.utils.to_categorical(labels)[:-1,:]

    dataset = tf.data.Dataset.from_tensor_slices(
        (windows.astype(np.float32), labels[:-1])
    ).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.shuffle(labels.size, reshuffle_each_iteration=True).batch(
        batch_size
    )
    pae, encoder = create_parallel_aes(
        window_size_per_ae, intermediate_dim, latent_dim, nr_shared
    )
    # pae.summary()
    # encoder.summary()
    # centers = tf.random.uniform(shape=(np.unique(labels).size, nr_shared))
    start_time = time.time()

    @tf.function
    def compute_mse_loss(x_in, x_dec):
        mse_loss = tf.reduce_mean(tf.square(x_dec - x_in))
        return mse_loss

    for epoch in range(nr_epochs):
        epoch_loss_avg = (
            tf.keras.metrics.Mean()
        )  # Keeping track of the training loss
        # print("==== Epoch {}/{} ====".format(epoch + 1, nr_epochs))

        for batch in dataset:
            x = batch[0]
            y = batch[1]

            with tf.GradientTape() as tape:  # Forward pass
                # compute losses
                x_decoded = pae(x, training=True)
                features = encoder(x, training=True)[:, :, :nr_shared]
                mse_loss = compute_mse_loss(x, x_decoded)
                if batch_all:
                    triplet_loss = batch_all_triplet_loss(
                        y, features, margin, squared
                    )[0]
                else:
                    triplet_loss = batch_hard_triplet_loss(
                        y, features, margin, squared
                    )
                loss = mse_loss + loss_weight_triplet * triplet_loss

            # collect trainable parameters
            pae_para = pae.trainable_variables
            trainable_para = pae_para
            grad = tape.gradient(loss, trainable_para)  # Backpropagation
            optimizer.apply_gradients(
                zip(grad, trainable_para)
            )  # Update network weights
            epoch_loss_avg(loss)

        # windows = ori_windows
        # dataset = tf.data.Dataset.from_tensor_slices((windows.astype(np.float32), labels)).prefetch(
        #     tf.data.AUTOTUNE)
        # dataset = dataset.shuffle(labels.size, reshuffle_each_iteration=True).batch(batch_size)
        loss_train[epoch] = epoch_loss_avg.result()
        # print("---- Training ----")
        # print("Loss  =  {0:.4e}".format(loss_train[epoch]))

        # if (epoch + 1) % 10 == 0:
        #     end_time = time.time()
            # print("time cost: ", end_time - start_time, "s")

    encoded_windows = encoder.predict(windows)
    shared_encoded_windows = np.concatenate(
        (
            encoded_windows[:, 0, :nr_shared],
            encoded_windows[-1:, 1, :nr_shared],
        ),
        axis=0,
    )
    return shared_encoded_windows
