from os import path

import pandas as pd
from dvc.repo import Repo
from sklearn.svm import SVC

from iris_classifiers.utils import add_dvc, save_to_onnx

DATA_PATH = "../data"
MODEL_PATH = "../model_repository/sklearn-onnx/1"
REPO_PATH = "../"


def main():
    repo = Repo(REPO_PATH)
    repo.pull()

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

    model_name = "model.onnx"
    save_to_onnx(model, X_train[:1], MODEL_PATH, model_name)
    add_dvc(MODEL_PATH, model_name)


if __name__ == "__main__":
    main()
