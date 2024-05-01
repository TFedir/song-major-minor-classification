import logging
import sys
from joblib import dump
from parse_data import load_tracks
from sklearn.linear_model import LogisticRegression
from training import train


def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.log(logging.INFO, "Loading dataset")
    features, targets,_ = load_tracks()
    logging.log(logging.INFO, "Fitting model")
    model = LogisticRegression()
    model = train(
        model, features, targets, True, "./src/models/logistic_regression.joblib"
    )


if __name__ == "__main__":
    main()
