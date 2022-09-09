import contextlib
import os

import numpy as np
import pandas as pd
from tsfuse.computation import extract
from tsfuse.data import Collection


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


@supress_stdout
def extract_features(X, C, ws, transformers="minimal"):

    C = C.copy()

    transformers = transformers

    ts_before = []
    ts_after = []
    for i in C.index:
        t = C.loc[i, "time"]
        b = X[C.loc[i, "sequence"]][(t - ws): t]
        a = X[C.loc[i, "sequence"]][t: (t + ws)]
        if (len(b) == ws) and (len(a) == ws):
            ts_before.append(b)
            ts_after.append(a)
        else:
            C = C.drop(i)

    if (len(ts_before) == 0) or (len(ts_after) == 0):
        assert len(C) == 0
        return C

    ts_before = np.array(ts_before)
    ts_after = np.array(ts_after)

    x_before = {
        "S{}".format(i + 1): Collection.from_array(ts_before[:, :, i : i + 1])
        for i in range(ts_before.shape[2])
    }
    x_after = {
        "S{}".format(i + 1): Collection.from_array(ts_after[:, :, i : i + 1])
        for i in range(ts_after.shape[2])
    }

    features_before = extract(x_before, transformers=transformers)
    features_after = extract(x_after, transformers=transformers)
    columns = [
        c for c in features_before.columns if c in features_after.columns
    ]
    features_before = features_before.loc[:, columns].astype(float)
    features_after = features_after.loc[:, columns].astype(float)

    features_difference = features_after - features_before

    features = features_difference
    features.index = C.index

    return C.join(features, how="inner")
