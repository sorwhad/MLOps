from os import makedirs, path, system

from skl2onnx import to_onnx


def save_to_onnx(model, types_like, model_path, model_name):
    if not path.exists(model_path):
        makedirs(model_path)

    onx = to_onnx(model, types_like)
    onx.ir_version = 8

    with open(path.join(model_path, model_name), "wb") as f:
        f.write(onx.SerializeToString())


def add_dvc(model_path, model_name):
    print("Adding model to dvc...")
    system(f"dvc add {path.join(model_path, model_name)}")
    system(f"git add {path.join(model_path, model_name + '.dvc')}")
    system(f"git add {path.join(model_path, '.gitignore')}")
    print("Done!")
