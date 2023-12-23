from functools import lru_cache
from os import path

import numpy as np
import pandas as pd
import tritonclient
from tritonclient.utils import np_to_triton_dtype

from iris_classifiers.utils import accuracy

DATA_PATH = "./data"


@lru_cache
def get_client():
    return tritonclient.http.InferenceServerClient(url="0.0.0.0:8500")


def call_triton_on_test_data(test_data):
    triton_client = get_client()

    input_test_data = tritonclient.http.InferInput(
        name="X", shape=test_data.shape, datatype=np_to_triton_dtype(test_data.dtype)
    )
    input_test_data.set_data_from_numpy(test_data, binary_data=True)

    infer_output = tritonclient.http.InferRequestedOutput("label", binary_data=True)
    query_resoponse = triton_client.infer(
        "sklearn-onnx", [input_test_data], outputs=[infer_output]
    ).as_numpy("label")[0]

    return query_resoponse


def test_shapes(y_true, y_hat):
    assert y_true.shape == y_hat.shape


def test_accuracy(y_true, y_hat):
    assert accuracy(y_hat, y_true) >= 0.8


def main():
    X_test = (
        pd.read_csv(path.join(DATA_PATH, "test_data.csv"), index_col=0)
        .to_numpy()
        .astype(np.float64)
    )
    y_true = (
        pd.read_csv(path.join(DATA_PATH, "test_target.csv"), index_col=0)
        .to_numpy()
        .reshape(-1)
    )
    y_hat = call_triton_on_test_data(X_test)

    test_shapes(y_true, y_hat)
    test_accuracy(y_hat, y_true)


if __name__ == "__main__":
    main()
