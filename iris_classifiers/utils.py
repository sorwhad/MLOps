from os import makedirs, path

from skl2onnx import to_onnx


def save_to_onnx(model, types_like, model_path, model_name):
    if not path.exists(model_path):
        makedirs(model_path)

    onx = to_onnx(model, types_like)
    with open(path.join(model_path, f"{model_name}.onnx"), "wb") as f:
        f.write(onx.SerializeToString())
