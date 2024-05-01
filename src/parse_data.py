import pandas
import logging
from sklearn import preprocessing
import numpy as np
import seaborn as sns

def load_tracks(path="./src/dane_v2/tracks.jsonl", drop_low_quality=False) -> (np.array, np.array, pandas.Index):
    tracks = pandas.read_json(path_or_buf="./src/dane_v2/tracks.jsonl", lines=True)

    logging.log(logging.INFO, "Preparing data")
    # remove rows without labels
    mask = tracks["mode"].isin([0, 1])
    tracks = tracks[mask]

    unused_columns = ["id", "name", "release_date", "id_artist"]
    if drop_low_quality:
        unused_columns+=["time_signature","explicit"]
    # remove unused columns
    tracks = tracks.drop(columns=unused_columns)
    #c = tracks.corr(method='spearman')
    #sns.heatmap(c, annot = True, cmap = 'twilight')
    for_training = tracks.drop(columns="mode")
    X = np.array(for_training)
    # normalise
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    return X, np.array(tracks["mode"]), for_training.columns