import os
from os import path

import joblib
import pandas as pd
from sklearn.svm import SVC

DATA_PATH = "../data"
MODEL_PATH = "../model"


def main():
    X_train = pd.read_csv(
        path.join(DATA_PATH, "train_data.csv"), index_col=0
    ).to_numpy()
    y_train = (
        pd.read_csv(path.join(DATA_PATH, "train_target.csv"), index_col=0)
        .to_numpy()
        .reshape(-1)
    )
    model = SVC()
    model.fit(X_train, y_train)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    joblib.dump(model, path.join(MODEL_PATH, "model.pkl"))


if __name__ == "__main__":
    main()
