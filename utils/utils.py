import os
import glob
import numpy as np
import pandas as pd
from utils.features import extract_features
from utils.tire import run_tire

def load_data(dir):
    """
    Load data from disk under the path OCSVM/data/...

    Args:
        dir: name of processing dataset
    Returns:
        X: loaded dataset in a dict (one ndarray for each sequence)
    """
    X = dict()
    cp = pd.DataFrame(columns=["sequence", "time"])
    base = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    dir = os.path.join(base, "data", dir)
    p = os.path.join(dir, f"*_data.txt")
    for filepath in sorted(glob.glob(p)):
        sequence = filepath[len(dir)+1 : -9]
        # Get time series data
        if ("hasc" in sequence) or ("eeg" in sequence) or ("UCI" in sequence):
            ts = pd.read_csv(filepath, header=None, sep="\t").values
        else:
            ts = pd.read_csv(filepath, header=None, sep=" ").values
        X[sequence] = ts
        # Get ground truth splits (for queries)
        regimes = pd.read_csv(
            filepath.replace("data.txt", "labels.txt"), header=None, sep=" "
        ).values.flatten()
        splits = list(np.where(regimes[1:] != regimes[:-1])[0])
        cp = cp.append(
            pd.DataFrame(
                {
                    "sequence": sequence,
                    "time": splits,
                }
            )
        )
        cp.index = np.arange(len(cp))
    return X, cp


def get_budget_and_ws(dataset):
    """
    Getting the budget and window size for the given dataset's name

    Args:
        dataset: the name of dataset
    Return:
        budget: the maximal number of queries
        ws: the length of window size
    """
    if (dataset == "mean") or (dataset == "var") or (dataset == "gauss"):
        return 720, 35, 10
    elif dataset == "ar":
        return 720, 35, 10
    elif dataset == "eeg":
        return 100, 200, 10
    elif dataset == "bee":
        return 180, 15, 5
    elif dataset == "baby":
        return 92, 15, 2
    elif dataset == "UCI":
        return 180, 8, 5


def TIRE(X, ws, epochs=200, seed=None):
    """
    Training the TIRE model for the initialization of algorithm

    Args:
        X: dataset to be processed
        ws: window size
        epochs: number of training epochs
        seed: random seed
    Returns:
        F: features of all samples in the given dataset
        C: features of candidates change points found by TIRE model
    """
    C = pd.DataFrame(columns=["sequence", "time", "change_point_score", "certainty"])
    F = pd.DataFrame(columns=["sequence", "time", "change_point_score", "certainty"])
    for di in X:
        print("TIRE" + "-" * 40 + di + "-" * 40)
        ts = X[di]
        cps = run_tire(ts, ws, nr_epochs=epochs, seed=seed)
        time = np.arange(len(cps))[cps > 0]
        time_all = np.arange(len(cps))
        cps = cps[cps > 0]
        for i, t in enumerate(time_all):
            if t in time:
                idx = np.argwhere(time == t)[0][0]
                C.loc[len(C), :] = [di, t, cps[idx], 0.0]
                F.loc[len(F), :] = [di, t, cps[idx], 0.0]
            else:
                F.loc[len(F), :] = [di, t, 0.0, 0.0]
    return extract_features(X, F, ws), extract_features(X, C, ws)


def get_all_features(X, ws):
    """
    Get all features corresponding to each sample

    Args:
        X: dataset to be processed
        ws: window size
    Returns:
        F: features of all samples in the given dataset
    """
    F = pd.DataFrame(columns=["sequence", "time", "change_point_score", "certainty"])
    for di in X:
        ts = X[di]
        time_all = np.arange(len(ts))
        for i, t in enumerate(time_all):
            F.loc[len(F), :] = [di, t, 0.0, 0.0]
    return extract_features(X, F, ws)


def remove_change_samples(F, ws):
    """
    Remove the samples in F that near to the candidate change points. Only use no-change samples to train the OCSVM.

    Args:
        F: features of all time samples in the give dataset
        ws: window size
    Returns:
        training set (without change samples) for fitting the OCSVM
    """
    sequences = sorted(list(F['sequence'].unique()))
    training_samples = pd.DataFrame(columns=F.columns.values)
    for seq in sequences:
        current_set = F.loc[F["sequence"] == seq]
        current_set = current_set.set_index(current_set["time"].values)
        current_indice = current_set["time"].values
        current_changes_indice = current_set.loc[current_set["change_point_score"] > 0].index.values
        remove_patch = np.arange(-ws, ws)
        remove_indice = sorted(set((np.matmul(np.expand_dims(current_changes_indice, 1),
                                  np.expand_dims(np.ones_like(remove_patch), 0)) + remove_patch).flatten()))
        remain_set = current_set[[i not in remove_indice for i in current_indice]]
        training_samples = pd.concat([training_samples, remain_set], ignore_index=True)
    return training_samples

def matched_filter(signal, window_size):
    """
    Matched filter for dissimilarity measure smoothing (and zero-delay weighted moving average filter for shared feature smoothing)

    Args:
        signal: input signal
        window_size: window size used for CPD
    Returns:
        filtered signal
    """
    mask = np.ones((2 * window_size + 1,))
    for i in range(window_size):
        mask[i] = i / (window_size ** 2)
        mask[-(i + 1)] = i / (window_size ** 2)
    mask[window_size] = window_size / (window_size ** 2)

    signal_out = np.zeros(np.shape(signal))

    if len(np.shape(signal)) > 1:  # for both TD and FD
        for i in range(np.shape(signal)[1]):
            signal_extended = np.concatenate((signal[0, i] * np.ones(window_size), signal[:, i], signal[-1, i] *
                                              np.ones(window_size)))
            signal_out[:, i] = np.convolve(signal_extended, mask, 'valid')
    else:
        signal = np.concatenate((signal[0] * np.ones(window_size), signal, signal[-1] * np.ones(window_size)))
        signal_out = np.convolve(signal, mask, 'valid')

    return signal_out


def precision_score(splits_true, splits_pred, threshold):
    """
    Compute the precision value in current sequence

    Args:
        splits_true: set of ground-truth change points in current sequence
        splits_pred: set of detected candidate change points in current sequence
        threshold: the tolerance to recognize a candidate change point is detected correctly
    Returns:
        the precision value of current sequence
    """
    tp = 0
    tp_set = []
    end = 0
    if (len(splits_true) == 0) or (len(splits_pred) == 0):
        return 0
    for gt in splits_true:
        splits_pred_tail = splits_pred[np.where(splits_pred > end)[0]]
        for p in splits_pred_tail:
            if np.abs(gt - p) <= threshold:
                end = p
                tp = tp + 1
                tp_set.append(gt)
                break
    return tp/len(splits_pred)


def recall_score(splits_true, splits_pred, threshold):
    """
    Compute the precision value in current sequence

    Args:
        splits_true: set of ground-truth change points in current sequence
        splits_pred: set of detected candidate change points in current sequence
        threshold: the tolerance to recognize a candidate change point is detected correctly
    Returns:
        the precision value of current sequence
    """
    tp = 0
    tp_set = []
    end = 0
    if (len(splits_true) == 0) or (len(splits_pred) == 0):
        return 0
    for gt in splits_true:
        splits_pred_tail = splits_pred[np.where(splits_pred > end)[0]]
        for p in splits_pred_tail:
            if np.abs(gt - p) <= threshold:
                end = p
                tp = tp + 1
                tp_set.append(gt)
                break
    return tp/len(splits_true)


def print_metric(cp, found_peaks, ws, title=None, save_results=False):
    """
    Print the averaged metrics (precision, recall, f1-score) over all sequences in the processed dataset

    Args:
        cp: location of ground-truth change points
        found_peaks: set of detected candidate change points
        ws: window size
        title: information to show in the title
        save_results: control whether return metric values
    Returns:
        None (if save_results is False) or metrics value (otherwise)
    """
    sequences = sorted(list(cp['sequence'].unique()))

    precision = np.mean([
        precision_score(
            cp.loc[cp['sequence'] == s, 'time'].values,
            found_peaks.loc[found_peaks['sequence'] == s, 'time'].values,
            ws
        )
        for s in sequences
    ])
    recall = np.mean([
        recall_score(
            cp.loc[cp['sequence'] == s, 'time'].values,
            found_peaks.loc[found_peaks['sequence'] == s, 'time'].values,
            ws
        )
        for s in sequences
    ])
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else np.nan
    if title is not None:
        print("-" * 40 + title + "-" * 40)
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1 score: " + str(f1))
    print()
    if save_results:
        return precision, recall, f1


def least_certain_candidate(score, peaks, Q, ws):
    """
    Find the candidate change point with lowest certainty to ask queries

    Args:
        score: the score that measure how likely a time sample should be recognized as a change point
        peaks: set of detected candidate change points detected in the target dataset
        Q: the set of queried points
    Returns:
        the index of found candidate change point that have lowest score
    """
    all_index_set = []
    for _, _, element in Q:
        index_set = np.arange(element - ws, element + ws + 1).tolist()
        all_index_set.extend(index_set)
    not_queried = [i for i in peaks if i not in all_index_set]
    not_queried_score = score[not_queried]
    if len(not_queried) > 0:
        return not_queried[np.argmin(not_queried_score)]
    else:
        return np.random.choice(peaks, size=1)[0]


def random_candidate(peaks, Q):
    """
    Find the candidate change point randomly to ask queries

    Args:
        peaks: set of detected candidate change points detected in the target dataset
        Q: the set of queried points
    Returns:
        the index of found candidate change point that have lowest score
    """
    not_queried = [i for i in peaks if i not in [q for _, _, q in Q]]
    if len(not_queried) > 0:
        return np.random.choice(not_queried, size=1)[0]
    else:
        return np.random.choice(peaks, size=1)[0]


def ordered_candidate(peaks, Q):
    """
    Find the candidate change point randomly to ask queries

    Args:
        peaks: set of detected candidate change points detected in the target dataset
        Q: the set of queried points
    Returns:
        the index of found candidate change point that have lowest score
    """
    not_queried = [i for i in peaks if i not in [q for _, _, q in Q]]
    if len(not_queried) > 0:
        return not_queried[0]
    else:
        return np.random.choice(peaks, size=1)[0]


def query_tire(q, F, cp, ws):
    """
    Simulate the query process

    Args:
        q: the index of time sample that a query is needed
        F: features of all time samples in the target dataset
        cp: location of ground-truth change points
        ws: window size
    Returns:
        query result
        sequence that contains the current query
        time step of current query (if query=False) or time step of corresponding GT (if query=True)
    """
    q_gts = []
    q_feature = F.loc[q]
    q_sequence = q_feature["sequence"]
    q_time = q_feature["time"]
    gt_set = cp.loc[cp['sequence'] == q_sequence, 'time'].to_frame()
    for idx_gt, gt in gt_set.iterrows():
        if np.abs(q_time - gt.values) <= ws:
            q_gts.append(gt.values[0])
    if len(q_gts) == 0:
        return False, q_sequence, q_time
    else:
        return True, q_sequence, q_gts


def query(q, F, cp, ws):
    """
    Simulate the query process

    Args:
        q: the index of time sample that a query is needed
        F: features of all time samples in the target dataset
        cp: location of ground-truth change points
        ws: window size
    Returns:
        query result
        sequence that contains the current query
        time step of current query (if query=False) or time step of corresponding GT (if query=True)
    """
    q_gts = []
    q_feature = F.loc[q]
    q_sequence = q_feature["sequence"]
    q_time = q_feature["time"]
    gt_set = cp.loc[cp['sequence'] == q_sequence, 'time'].to_frame()
    for idx_gt, gt in gt_set.iterrows():
        if np.abs(q_time - gt.values) <= ws:
            return True, q_sequence, gt.values[0]
    return False, q_sequence, q_time


def adjust_with_queries(peaks, Q, ws, F):
    """
    Manually adjust detection results based on queries

    Args:
        peaks: set of candidate change points found by OCSVM
        Q: the set of queried points
        ws: window size
        F: features of all time samples in the target dataset
    Returns:
        adjusted detection results
    """
    Q_array = np.array(Q)
    Q_array = Q_array[Q_array[:, 1].argsort()]
    adjusted_peaks = peaks
    boundary = ws

    for q, a, gt_time in Q_array:
        q_feature = F.loc[q]
        q_sequence = q_feature["sequence"]
        current_seq = F.loc[F["sequence"] == q_sequence]
        index_set = current_seq.index.values
        if not a:
            if (q - boundary) in index_set:
                start_point = q - boundary
            else:
                start_point = index_set[0]
            if (q + boundary) in index_set:
                end_point = q + boundary
            else:
                end_point = index_set[-1]
            index_set = np.arange(start_point, end_point + 1).tolist()
            adjusted_peaks = [i for i in adjusted_peaks if i not in index_set]
        else:
            if (gt_time - boundary) in index_set:
                start_point = gt_time - boundary
            else:
                start_point = index_set[0]
            if (gt_time + boundary) in index_set:
                end_point = gt_time + boundary
            else:
                end_point = index_set[-1]
            index_set = np.arange(start_point, end_point + 1).tolist()
            adjusted_peaks = [i for i in adjusted_peaks if i not in index_set]
            adjusted_peaks.append(gt_time)
    adjusted_peaks = np.array(adjusted_peaks)
    adjusted_peaks.sort()
    return adjusted_peaks


def adjust_with_queries_half(peaks, Q, ws, F):
    """
    Manually adjust detection results based on queries

    Args:
        peaks: set of candidate change points found by OCSVM
        Q: the set of queried points
        ws: window size
        F: features of all time samples in the target dataset
    Returns:
        adjusted detection results
    """
    Q_array = np.array(Q)
    Q_array = Q_array[Q_array[:, 1].argsort()]
    adjusted_peaks = peaks
    boundary = ws

    for q, a, gt_time in Q_array:
        q_feature = F.loc[q]
        q_sequence = q_feature["sequence"]
        current_seq = F.loc[F["sequence"] == q_sequence]
        index_set = current_seq.index.values
        if not a:
            if (q - boundary) in index_set:
                start_point = q - boundary
            else:
                start_point = index_set[0]
            if (q + boundary) in index_set:
                end_point = q + boundary
            else:
                end_point = index_set[-1]
            index_set = np.arange(start_point, end_point + 1).tolist()
            adjusted_peaks = [i for i in adjusted_peaks if i not in index_set]
        else:
            if (q - boundary) in index_set:
                start_point = q - boundary
            else:
                start_point = index_set[0]
            if (q + boundary) in index_set:
                end_point = q + boundary
            else:
                end_point = index_set[-1]
            index_set = np.arange(start_point, end_point + 1).tolist()
            adjusted_peaks = [i for i in adjusted_peaks if i not in index_set]
            adjusted_peaks.append(q)
    adjusted_peaks = np.array(adjusted_peaks)
    adjusted_peaks.sort()
    return adjusted_peaks
