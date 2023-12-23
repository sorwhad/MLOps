from os import path

import numpy as np
import onnxruntime as rt
import pandas as pd
from sklearn.metrics import accuracy_score

DATA_PATH = "../data"
MODEL_PATH = "../model_repository/sklearn-onnx/1"
REPO_PATH = "../"


def print_metrics(y_hat, y_test):
    print("ACCURACY:", accuracy_score(y_hat, y_test))


def main():
    sess = rt.InferenceSession(
        path.join(MODEL_PATH, "model.onnx"), providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    X_test = pd.read_csv(path.join(DATA_PATH, "test_data.csv"), index_col=0).to_numpy()
    y_hat = sess.run([label_name], {input_name: X_test.astype(np.float64)})[0]

    y_test = (
        pd.read_csv(path.join(DATA_PATH, "test_target.csv"), index_col=0)
        .to_numpy()
        .reshape(-1)
    )

    # y_hat = loaded_model.predict(X_test)
    pd.DataFrame(y_hat).to_csv(path.join(DATA_PATH, "prediction_target.csv"))
    print_metrics(y_hat, y_test)


if __name__ == "__main__":
    main()
