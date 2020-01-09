import logging
import os

import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

from core import RetinaNet_Except_Anchor, AnchorNet, export_block_for_cplusplus
from core import check_onnx
from core import testdataloader

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

def export(path="weights",
           newpath="exportweights",
           load_name="768_768_ADAM_PRES_18",
           load_period=1,
           target_size=(768, 768),
           anchor_sizes=[32, 64, 128, 256, 512],
           anchor_size_ratios=[1, pow(2, 1 / 3), pow(2, 2 / 3)],
           anchor_aspect_ratios=[0.5, 1, 2],
           anchor_box_offset=(0.5, 0.5),
           anchor_box_clip=False,
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
    net = RetinaNet_Except_Anchor(version=version,
                                  input_size=target_size,
                                  anchor_size_ratios=anchor_size_ratios,
                                  anchor_aspect_ratios=anchor_aspect_ratios,
                                  num_classes=test_dataset.num_class)
    net.load_parameters(params, allow_missing=True,
                        ignore_extra=True)

    '''
    get_fpn_resnet AnchorNet에서 선언한 순간 내부의 auxiliary param으로 인식하기 때문에 반드시 initialization을 해줘야 하는데,
    (외부에서 선언해서 보내면 이럴 필요없고 net.export만 수정해주면 된다.
    defer init을 쓴 상태라서 실제 forward를 한번 해줘야 한다.(anchornet.forward(mx.nd.ones(shape=(1, 3) + target_size, ctx=ctx))) 
    현재는 밖에서 보내면, 순간 내부의 auxiliary param으로 인식하지는 않아서, forward를 할 필요는 없다.
    어쨌든 두 방법 모두 사용하지 않는 변수가 생기기 때문에,  net.export(내부 코드)의 assert name in aux_names 부분을 주석 처리 해야한다.
    단 위와 같이 저장 한 경우, json, param을 불러 올 때, 아래와 같이 
    allow_missing=True, ignore_extra=True 인자를 gluon.SymbolBlock.imports(내부코드)를 수정해야 한다.
    ret.collect_params().load 함수에 allow_missing=True, ignore_extra=True 를 주면 된다 
    net = gluon.SymbolBlock.imports(sym, ['data'], params, ctx=ctx, allow_missing=True, ignore_extra=True)
    --> 인자로 주게 만들어 놓지... 

    < 상세 설명 >
    target_size에 맞는 anchor를 얻기 위해서는 아래의 AnchorNet에서 get_fpn_resnet 네트워크를 또 한번 호출해야 한다.
    ->  이렇게 되면 새롭게 호출하는 get_fpn_resnet가 새로 저장하는 json에 추가적으로 써지고, 파라미터도 저장이 되기 때문에 비효율적이다.(해결책? 아직은...)
        이에 따라 또다른 문제가 생기는데(정확히 말해서 net.export에 내가 말하고자 하는 기능이 고려가 안되있다.), 

        현재 net.export는 할당한 파라미터들을 forward 에서 다 사용하지 않으면, AssertionError를 발생시킨다.

        실제 연산에 사용되는 파라미터만 저장하려고 하기 떄문에 이러한 오류가 발생하는데, 그냥 다 저장하게 net.export의
        내부 코드 한줄을 수정하면 된다. 
        이렇게 저장한 json, param을 불러 올때는 아래와 같이 불러와야 하는데, allow_missing=True, ignore_extra=True 는 
        본인이 직접 추가한 것이다.(gluon.SymbolBlock.imports 에는 allow_missing, ignore_extra 인자가 고려 되어 있지 않다.)
        gluon.SymbolBlock.imports(sym, ['data'], params, ctx=ctx, allow_missing=True, ignore_extra=True)
        아래와 같이 gluon.SymbolBlock.imports을 수정하면 된다. 

    def imports(symbol_file, input_names, param_file=None, ctx=None, allow_missing=True, ignore_extra=True):
        sym = symbol.load(symbol_file)
        if isinstance(input_names, str):
            input_names = [input_names]
        if param_file is None:
            # Get a valid type inference by using fp32
            inputs = [symbol.var(i, dtype=mx_real_t) for i in input_names]
        else:
            # Do not specify type, rely on saved params type instead
            inputs = [symbol.var(i) for i in input_names]
        ret = SymbolBlock(sym, inputs)
        if param_file is not None:
            # allow_missing=True, ignore_extra=True 추가 함.
            ret.collect_params().load(param_file, ctx=ctx, cast_dtype=True, dtype_source='saved', 
            allow_missing=allow_missing, ignore_extra=ignore_extra) 
        return ret

        따라서 안쓰는 파라미터를 저장하지 않으면 net.export 내부를 수정해야 한다.(아래 설명) 
    '''
    anchornet = AnchorNet(net=net, version=version, target_size=target_size,
                          anchor_sizes=anchor_sizes,
                          anchor_size_ratios=anchor_size_ratios,
                          anchor_aspect_ratios=anchor_aspect_ratios,
                          box_offset=anchor_box_offset,
                          anchor_box_clip=anchor_box_clip)

    new_weight_path = os.path.join(newpath, load_name)
    if not os.path.exists(new_weight_path):
        os.makedirs(new_weight_path)

    newname = str(target_size[0]) + "_" + str(target_size[1]) + "_" + temp[2] + "_" + temp[3] + "_" + temp[4]
    sym_pre = os.path.join(new_weight_path, f'{newname}_pre-symbol.json')
    params_pre = os.path.join(new_weight_path, f'{newname}_pre-{load_period:04d}.params')
    onnx_pre_file_path = os.path.join(new_weight_path, f"{newname}_pre.onnx")

    try:
        '''
        def export(self, path, epoch=0, remove_amp_cast=True):
        """Export HybridBlock to json format that can be loaded by
        `SymbolBlock.imports`, `mxnet.mod.Module` or the C++ interface.

        .. note:: When there are only one input, it will have name `data`. When there
                  Are more than one inputs, they will be named as `data0`, `data1`, etc.

        Parameters
        ----------
        path : str
            Path to save model. Two files `path-symbol.json` and `path-xxxx.params`
            will be created, where xxxx is the 4 digits epoch number.
        epoch : int
            Epoch number of saved model.
        """
        if not self._cached_graph:
            raise RuntimeError(
                "Please first call block.hybridize() and then run forward with "
                "this block at least once before calling export.")
        sym = self._cached_graph[1]
        sym.save('%s-symbol.json'%path, remove_amp_cast=remove_amp_cast)

        arg_names = set(sym.list_arguments())
        aux_names = set(sym.list_auxiliary_states())
        arg_dict = {}
        for name, param in self.collect_params().items():
            if name in arg_names:
                arg_dict['arg:%s'%name] = param._reduce()
            else:
                #assert name in aux_names #  여기 주석 처리 해야함 
                arg_dict['aux:%s'%name] = param._reduce()
        ndarray.save('%s-%04d.params'%(path, epoch), arg_dict)
        '''
        export_block_for_cplusplus(path=os.path.join(new_weight_path, f"{newname}_pre"),
                                   block=anchornet,
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
    export(path="weights", newpath="exportweights",
           load_name="512_512_ADAM_PRES_18",
           load_period=1,
           target_size=(768, 768),
           anchor_sizes=[32, 64, 128, 256, 512],
           anchor_size_ratios=[1, pow(2, 1 / 3), pow(2, 2 / 3)],
           anchor_aspect_ratios=[0.5, 1, 2],
           anchor_box_offset=(0.5, 0.5),
           anchor_box_clip=False,
           dtype=np.float32)
