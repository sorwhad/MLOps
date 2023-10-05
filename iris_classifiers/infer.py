from os import path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

DATA_PATH = "../data"
MODEL_PATH = "../model"


def print_metrics(y_hat, y_test):
    print("ACCURACY:", accuracy_score(y_hat, y_test))


if __name__ == "__main__":
    loaded_model = joblib.load(path.join(MODEL_PATH, "model.pkl"))

    X_test = pd.read_csv(path.join(DATA_PATH, "test_data.csv"), index_col=0).to_numpy()
    y_test = (
        pd.read_csv(path.join(DATA_PATH, "test_target.csv"), index_col=0)
        .to_numpy()
        .reshape(-1)
    )

    # X_test = pd.read_csv(path.join(DATA_PATH, "test_data.csv"))
    # y_test = pd.read_csv(path.join(DATA_PATH, "test_target.csv"))

    y_hat = loaded_model.predict(X_test)
    pd.DataFrame(y_hat).to_csv(path.join(DATA_PATH, "prediction_target.csv"))
    print_metrics(y_hat, y_test)
