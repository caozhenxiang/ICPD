import os
import pickle
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from utils.utils import *
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

# ------------------- settings ------------------- #
seed = None
epochs = 200
standardscale = True
gamma = 2.5
nu = 1e-3
boundary_factor = 1.0
dataset = "mean"   # "var", "ar", "gauss", "mean", "bee" , "UCI", "baby"
query_strategy = ["lc", "round"]     # define the query strategy
budget, ws, r = get_budget_and_ws(dataset)
data, cp = load_data(dataset)
sequences = sorted(list(cp['sequence'].unique()))

# ------------------- Initilization ------------------- #
base = os.getcwd()
F, C = TIRE(data, ws, epochs, seed)
candidates = C.iloc[:, :4]
precision, recall, f1 = print_metric(cp, C, ws, title="Baseline", save_results=True)

F.reset_index(drop=True, inplace=True)
found_set = F.loc[F["change_point_score"] > 0]
init_peaks = found_set.index.values

# ------------------- First round ------------------- #
boundary = int(ws * boundary_factor)
training_samples = remove_change_samples(F, ws)
clf = OneClassSVM(gamma=gamma, nu=nu)
if standardscale:
    ss = StandardScaler()
    training_set = ss.fit_transform(training_samples.iloc[:, 4:].values)
    clf.fit(training_set)
    X = F.iloc[:, 4:].values
    score = clf.score_samples(ss.transform(X))
else:
    clf.fit(training_samples.iloc[:, 4:].values)
    X = F.iloc[:, 4:].values
    score = clf.score_samples(X)
score = matched_filter(score, int(1.0*ws))
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
    q = least_certain_candidate(score, adj_peaks, Q, boundary)
    a, q_sequence, q_time = query(q, F, cp=cp, ws=ws)
    current_seq = F.loc[F["sequence"] == q_sequence]
    current_index_set = current_seq.index.values
    if q_time < ws:
        q_gt = 0
    else:
        q_gt = current_seq.loc[current_seq["time"] == q_time].index.values[0]
    Q.append((q, a, q_gt))

    # adjust the training set based on q(e)
    if a:
        if (q_gt - boundary) in current_index_set:
            start_point = q_gt - boundary
        else:
            start_point = current_index_set[0]
        if (q_gt + boundary) in current_index_set:
            end_point = q_gt + boundary
        else:
            end_point = current_index_set[-1]
        index_set = np.arange(start_point, end_point + 1).tolist()
        F.loc[index_set, "change_point_score"] = 0.0
        F.at[q_gt, "change_point_score"] = 0.555555555555
    else:
        if (q - boundary) in current_index_set:
            start_point = q - boundary
        else:
            start_point = current_index_set[0]
        if (q + boundary) in current_index_set:
            end_point = q + boundary
        else:
            end_point = current_index_set[-1]
        index_set = np.arange(start_point, end_point + 1).tolist()
        F.loc[index_set, "change_point_score"] = 0.0
    # retrain OCSVM
    if len(Q) % r == 0:
        training_samples = remove_change_samples(F, boundary)
        clf = OneClassSVM(gamma=gamma, nu=nu)
        if standardscale:
            ss = StandardScaler()
            training_set = ss.fit_transform(training_samples.iloc[:, 4:].values)
            clf.fit(training_set)
            X = F.iloc[:, 4:].values
            score = clf.score_samples(ss.transform(X))
        else:
            clf.fit(training_samples.iloc[:, 4:].values)
            X = F.iloc[:, 4:].values
            score = clf.score_samples(X)
        score = matched_filter(score, ws)
        score = score.max() - score
        peaks = find_peaks(score)[0]
        adj_peaks = adjust_with_queries(peaks, Q, boundary, F)
        found_peaks = F.iloc[peaks]
        adj_found_peaks = F.iloc[adj_peaks]
        precision, recall, f1 = print_metric(cp, found_peaks, ws, title=str(n) + "without adjust", save_results=True)
        adj_precision, adj_recall, adj_f1 = print_metric(cp, adj_found_peaks, ws, title=str(n) + "with adjust",
                                                         save_results=True)
    # update round
    n = n + 1