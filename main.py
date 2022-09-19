import os
import pickle
from sklearn.svm import OneClassSVM
from utils.utils import *
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

# ------------------- settings ------------------- #
seed = None
epochs = 200        # number of epochs for TIRE training
gamma = 2.5         # value of kernel coefficient in OCSVM
nu = 1e-3           # value of training error bound in OCSVM
dataset = "mean"    # name of dataset
budget = 720        # budget for maximal queries' number
ws = 35             # predefined window size
r = 10              # the round interval to retrain OCSVM model,
                    # i.e., the OCSVM is retrained after collecting each r queries

# ------------------- Initialization ------------------- #
# load data from disk
data, cp = load_data(dataset)
sequences = sorted(list(cp['sequence'].unique()))

# training TIRE model for the initialization
F, C = TIRE(data, ws, epochs, seed)
candidates = C.iloc[:, :4]
precision, recall, f1 = print_metric(cp, C, ws, title="Baseline", save_results=True)

# obtain the feature matrix from TSfuse system
F.reset_index(drop=True, inplace=True)
found_set = F.loc[F["change_point_score"] > 0]
init_peaks = found_set.index.values

# ------------------- First round ------------------- #
training_samples = remove_change_samples(F, ws)
X = F.iloc[:, 4:].values
clf = OneClassSVM(gamma=gamma, nu=nu)
clf.fit(training_samples.iloc[:, 4:].values)
score = clf.score_samples(X)
score = matched_filter(score, ws)
score = score.max() - score
peaks = find_peaks(score)[0]
peaks_first_round = peaks.copy()
adj_peaks = peaks.copy()
found_peaks = F.iloc[peaks]
precision, recall, f1 = print_metric(cp, found_peaks, ws, title="0", save_results=True)
candidates = found_peaks.iloc[:, :4]

# ------------------- Active learning ------------------- #
Q = []
n = 0
while len(Q) < budget:
    # find the element e to get q(e)
    q = least_certain_candidate(score, adj_peaks, Q, ws)
    a, q_sequence, q_time = query(q, F, cp=cp, ws=ws)
    current_seq = F.loc[F["sequence"] == q_sequence]
    current_index_set = current_seq.index.values
    if q_time < ws:
        q_gt = current_seq.iloc[:, 0].index.values[0]
    else:
        q_gt = current_seq.loc[current_seq["time"] == q_time].index.values[0]
    Q.append((q, a, q_gt))

    # adjust the training set based on q(e)
    if a:
        if (q_gt - ws) in current_index_set:
            start_point = q_gt - ws
        else:
            start_point = current_index_set[0]
        if (q_gt + ws) in current_index_set:
            end_point = q_gt + ws
        else:
            end_point = current_index_set[-1]
        index_set = np.arange(start_point, end_point + 1).tolist()
        F.loc[index_set, "change_point_score"] = 0.0
        F.at[q_gt, "change_point_score"] = 0.555555555555
    else:
        if (q - ws) in current_index_set:
            start_point = q - ws
        else:
            start_point = current_index_set[0]
        if (q + ws) in current_index_set:
            end_point = q + ws
        else:
            end_point = current_index_set[-1]
        index_set = np.arange(start_point, end_point + 1).tolist()
        F.loc[index_set, "change_point_score"] = 0.0

    # retrain OCSVM
    if len(Q) % r == 0:
        training_samples = remove_change_samples(F, ws)
        clf = OneClassSVM(gamma=gamma, nu=nu)
        clf.fit(training_samples.iloc[:, 4:].values)
        X = F.iloc[:, 4:].values
        score = clf.score_samples(X)
        score = matched_filter(score, ws)
        score = score.max() - score
        peaks = find_peaks(score)[0]
        adj_peaks = adjust_with_queries(peaks, Q, ws, F)
        adj_found_peaks = F.iloc[adj_peaks]
        adj_precision, adj_recall, adj_f1 = print_metric(cp, adj_found_peaks, ws, title=str(n) + "with adjust",
                                                         save_results=True)
    # update round number
    n = n + 1