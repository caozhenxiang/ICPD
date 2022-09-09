import numpy as np

from .postprocessing import *


def smoothened_dissimilarity_measures(
    encoded_windows=None, encoded_windows_fft=None, window_size=20
):
    """
    Calculation of smoothened dissimilarity measures

    Args:
        encoded_windows: TD latent representation of windows
        encoded_windows_fft:  FD latent representation of windows
        domain: TD/FD/both
        parameters: array with used parameters
        window_size: window size used
        par_smooth

    Returns:
        smoothened dissimilarity measures
    """
    if encoded_windows_fft is None:
        encoded_windows_both = encoded_windows
    elif encoded_windows is None:
        encoded_windows_both = encoded_windows_fft
    else:
        beta = np.quantile(distance(encoded_windows, window_size), 0.95)
        alpha = np.quantile(distance(encoded_windows_fft, window_size), 0.95)
        encoded_windows_both = np.concatenate(
            (encoded_windows * alpha, encoded_windows_fft * beta), axis=1
        )

    encoded_windows_both = matched_filter(
        encoded_windows_both, window_size
    )  # smoothing for shared features (9)
    distances = distance(encoded_windows_both, window_size)
    distances = matched_filter(
        distances, window_size
    )  # smoothing for dissimilarity (12)

    return distances


def change_point_score(distances, window_size):
    """
    Gives the change point score for each time stamp. A change point score > 0 indicates that a new segment starts at that time stamp.

    Args:
    distances: postprocessed dissimilarity measure for all time stamps
    window_size: window size used in TD for CPD

    Returns:
    change point scores for every time stamp (i.e. zero-padded such that length is same as length time series)
    """
    prominences = np.array(new_peak_prominences(distances)[0])
    prominences = prominences / np.amax(prominences)
    return np.concatenate(
        (np.zeros((window_size,)), prominences, np.zeros((window_size - 1,)))
    )
