import logging
import os

from core import Prediction
from core import export_block_for_cplusplus, PostNet
from mxnet import gluon

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def export(originpath="weights",
           newpath="jsonparamweights",
           load_name="480_640_ADAM_PCENTER_RES18",
           load_period=1,
           topk=200):

    prediction = Prediction(topk=topk)

    origin_weight_path = os.path.join(originpath, load_name)
    sym_path = os.path.join(origin_weight_path, f'{load_name}-symbol.json')
    param_path = os.path.join(origin_weight_path, f'{load_name}-{load_period:04d}.params')
    temp = load_name.split("_")

    new_weight_path = os.path.join(newpath, load_name)

    if not os.path.exists(new_weight_path):
        os.makedirs(new_weight_path)

    if os.path.exists(sym_path) and os.path.exists(param_path):
        logging.info(f"loading {os.path.basename(param_path)} weights\n")
        net = gluon.SymbolBlock.imports(sym_path,
                                        ['data'],
                                        param_path)
    else:
        raise FileExistsError

    # prepost
    postnet = PostNet(net=net, auxnet=prediction)

    try:
        export_block_for_cplusplus(path=os.path.join(new_weight_path, f"{load_name}_prepost"),
                                   block=postnet,
                                   data_shape=tuple((int(temp[0]), int(temp[1]))) + tuple((3,)),
                                   epoch=load_period,
                                   preprocess=True,  # c++ 에서 inference시 opencv에서 읽은 이미지 그대로 넣으면 됨
                                   layout='HWC',
                                   remove_amp_cast=True)
    except Exception as E:
        logging.error(f"json, param model export 예외 발생 : {E}")
    else:
        logging.info("json, param model export 성공")


if __name__ == "__main__":
    export(originpath="weights",
           newpath="jsonparamweights",
           load_name="480_640_ADAM_PCENTER_RES18",
           load_period=1,
           topk=100)
