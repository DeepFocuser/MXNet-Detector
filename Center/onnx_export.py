import logging
import os

import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

from core import check_onnx
from core import testdataloader

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

def export(path="weights",
           newpath="exportweights",
           load_name="480_640_ADAM_PCENTER_RES18",
           load_period=1,
           target_size=(768, 768),
           dtype=np.float32):
    try:
        _, test_dataset = testdataloader()
    except Exception:
        logging.info("The dataset does not exist")
        exit(0)

    weight_path = os.path.join(path, load_name)
    if not os.path.exists(weight_path):
        raise FileExistsError

    temp = load_name.split("_")
    new_weight_path = os.path.join(newpath, load_name)
    if not os.path.exists(new_weight_path):
        os.makedirs(new_weight_path)

    newname = str(target_size[0]) + "_" + str(target_size[1]) + "_" + temp[2] + "_" + temp[3] + "_" + temp[4]
    sym = os.path.join(weight_path, f'{newname}_prepost-symbol.json')
    params = os.path.join(weight_path, f'{newname}_prepost-{load_period:04d}.params')
    onnx_pre_file_path = os.path.join(new_weight_path, f"{newname}_prepost.onnx")


    try:
        onnx_mxnet.export_model(sym=sym, params=params,
                                input_shape=[tuple((1,)) + tuple(target_size) + tuple((3,))],
                                input_type=dtype,
                                onnx_file_path=onnx_pre_file_path, verbose=False)
    except Exception as E:
        logging.error(f"ONNX model export 예외 발생 : {E}")
    else:
        logging.info(f"ONNX model export 성공")

    try:
        check_onnx(onnx_pre_file_path)
        logging.info(f"{os.path.basename(onnx_pre_file_path)} saved completed")
    except Exception as E:
        logging.error(f"ONNX model check 예외 발생 : {E}")
    else:
        logging.info("ONNX model check completed")


if __name__ == "__main__":
    export(path="weights", newpath="exportweights",
           load_name="480_640_ADAM_PCENTER_RES18",
           load_period=1,
           target_size=(768, 768),
           dtype=np.float32)
