import logging
import os

import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

from core import YoloV3output, AnchorOffstNet, export_block_for_cplusplus
from core import check_onnx
from core import testdataloader

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def onnx_export(path="weights",
                newpath="exportweights",
                load_name="608_608_SGD_PDark_53",
                load_period=1,
                target_size=(768, 768),
                anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                         "middle": [(30, 61), (62, 45), (59, 119)],
                         "deep": [(116, 90), (156, 198), (373, 326)]},
                dtype=np.float32):
    try:
        _, test_dataset = testdataloader()
    except Exception:
        logging.info("The dataset does not exist")
        exit(0)

    weight_path = os.path.join(path, load_name)
    if not os.path.exists(weight_path):
        raise FileExistsError

    params = os.path.join(weight_path, f'{load_period}.params')

    temp = load_name.split("_")
    version = int(temp[-1])
    net = YoloV3output(Darknetlayer=version,
                       anchors=anchors,
                       num_classes=test_dataset.num_class)
    net.load_parameters(params, allow_missing=True,
                        ignore_extra=True)

    anchoroffstnet = AnchorOffstNet(net=net, version=version, anchors=anchors, target_size=target_size)

    new_weight_path = os.path.join(newpath, load_name)
    if not os.path.exists(new_weight_path):
        os.makedirs(new_weight_path)

    newname = str(target_size[0]) + "_" + str(target_size[1]) + "_" + temp[2] + "_" + temp[3] + "_" + temp[4]
    sym_pre = os.path.join(new_weight_path, f'{newname}_pre-symbol.json')
    params_pre = os.path.join(new_weight_path, f'{newname}_pre-{load_period:04d}.params')
    onnx_pre_file_path = os.path.join(new_weight_path, f"{newname}_pre.onnx")

    export_block_for_cplusplus(path=os.path.join(new_weight_path, f"{newname}_pre"),
                               block=anchoroffstnet,
                               data_shape=tuple(target_size) + tuple((3,)),
                               epoch=load_period,
                               preprocess=True,  # c++ 에서 inference시 opencv에서 읽은 이미지 그대로 넣으면 됨
                               layout='HWC',
                               remove_amp_cast=True)

    try:
        export_block_for_cplusplus(path=os.path.join(new_weight_path, f"{newname}_pre"),
                                   block=anchoroffstnet,
                                   data_shape=tuple(target_size) + tuple((3,)),
                                   epoch=load_period,
                                   preprocess=True,  # c++ 에서 inference시 opencv에서 읽은 이미지 그대로 넣으면 됨
                                   layout='HWC',
                                   remove_amp_cast=True)
    except Exception as E:
        logging.error(f"json, param model export 예외 발생 : {E}")
    else:
        logging.info("json, param model export 성공")

    try:
        onnx_mxnet.export_model(sym=sym_pre, params=params_pre,
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
    onnx_export(path="weights",
                newpath="exportweights",
                load_name="608_608_SGD_PDark_53",
                load_period=1,
                target_size=(768, 768),
                anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                         "middle": [(30, 61), (62, 45), (59, 119)],
                         "deep": [(116, 90), (156, 198), (373, 326)]},
                dtype=np.float32)
