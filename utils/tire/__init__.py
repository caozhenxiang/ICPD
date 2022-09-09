import numpy as np

from .evaluation import *
from .preprocessing import *
from .tire import *
from .utils import *
from .architectures import trip_semi_network


def run_tire(
    ts,
    ws,
    nr_epochs=200,
    domain="both",
    intermediate_dim_TD=30,
    latent_dim_TD=3,
    nr_shared_TD=2,
    K_TD=1,
    loss_weight_TD=0.001,
    intermediate_dim_FD=30,
    latent_dim_FD=3,
    nr_shared_FD=2,
    K_FD=1,
    loss_weight_FD=0.001,
    nfft=30,
    norm_mode="timeseries",
    seed=None,
):
    nr_ae_TD = K_TD + 1
    nr_ae_FD = K_FD + 1
    N = ts.shape[0]
    if ts.shape[1] == 1:
        ts = ts.flatten()
    else:
        ts = np.sqrt(np.sum(np.square(ts), axis=1))
    assert ts.shape[0] == N
    windows_TD = ts_to_windows(ts, 0, ws, 1)
    windows_TD = minmaxscale(windows_TD, -1, 1)
    windows_FD = calc_fft(windows_TD, nfft, norm_mode)
    shared_features_TD = train_AE(
        windows_TD,
        intermediate_dim_TD,
        latent_dim_TD,
        nr_shared_TD,
        nr_ae_TD,
        loss_weight_TD,
        nr_epochs,
        seed=seed,
    )
    shared_features_FD = train_AE(
        windows_FD,
        intermediate_dim_FD,
        latent_dim_FD,
        nr_shared_FD,
        nr_ae_FD,
        loss_weight_FD,
        nr_epochs,
        seed=seed,
    )
    dissimilarities = smoothened_dissimilarity_measures(
        shared_features_TD, shared_features_FD, ws
    )
    return change_point_score(dissimilarities, ws)


def run_stire(
    ts,
    ws,
    examples,
    nr_epochs=200,
    domain="both",
    intermediate_dim_TD=30,
    latent_dim_TD=3,
    nr_shared_TD=2,
    shared_loss_weight_TD=0.01,
    intermediate_dim_FD=30,
    latent_dim_FD=3,
    nr_shared_FD=2,
    shared_loss_weight_FD=0.01,
    nfft=30,
    norm_mode="timeseries",
    batch_all=False,
    margin=0.1,
    squared=True,
    seed=None,
):
    N = ts.shape[0]
    if ts.shape[1] == 1:
        ts = ts.flatten()
    else:
        ts = np.sqrt(np.sum(np.square(ts), axis=1))
    assert ts.shape[0] == N
    left_peaks = np.array(sorted(examples))
    windows_TD = ts_to_windows(ts, 0, ws, 1)
    windows_TD = minmaxscale(windows_TD, -1, 1)
    windows_FD = calc_fft(windows_TD, nfft, norm_mode)
    i = 0
    labels = np.zeros(windows_TD.shape[0], dtype=int) + len(left_peaks)
    for s, j in enumerate(left_peaks):
        labels[i:j] = s
        i = j
    network = trip_semi_network
    if domain == "TD":
        shared_features_TD = network.train_AE(
            windows_TD,
            labels,
            intermediate_dim_TD,
            latent_dim_TD,
            nr_shared_TD,
            shared_loss_weight_TD,
            batch_all,
            margin,
            squared,
            nr_epochs,
            seed=seed,
        )
        shared_features_FD = None
    elif domain == "FD":
        shared_features_TD = None
        shared_features_FD = network.train_AE(
            windows_FD,
            labels,
            intermediate_dim_FD,
            latent_dim_FD,
            nr_shared_FD,
            shared_loss_weight_FD,
            batch_all,
            margin,
            squared,
            nr_epochs,
            seed=seed,
        )
    else:
        shared_features_TD = network.train_AE(
            windows_TD,
            labels,
            intermediate_dim_TD,
            latent_dim_TD,
            nr_shared_TD,
            shared_loss_weight_TD,
            batch_all,
            margin,
            squared,
            nr_epochs,
            seed=seed,
        )
        shared_features_FD = network.train_AE(
            windows_FD,
            labels,
            intermediate_dim_FD,
            latent_dim_FD,
            nr_shared_FD,
            shared_loss_weight_FD,
            batch_all,
            margin,
            squared,
            nr_epochs,
            seed=seed,
        )
    dissimilarities = evaluation.smoothened_dissimilarity_measures(
        shared_features_TD, shared_features_FD, ws
    )
    return change_point_score(dissimilarities, ws)
