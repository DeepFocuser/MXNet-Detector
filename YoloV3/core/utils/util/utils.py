import os
import random
from collections import OrderedDict

import cv2
import mxnet as mx
import numpy as np
import onnx
from matplotlib import pyplot as plt
from mxnet.base import MXNetError
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential, Conv2D, Conv2DTranspose, BatchNorm, LeakyReLU
from onnx import checker

from core.model.backbone.Darknet import get_darknet


def check_onnx(onnx_path):
    '''
        Now we can check validity of the converted ONNX model by using ONNX checker tool.
        The tool will validate the model by checking if the content contains valid protobuf:
        If the converted protobuf format doesn’t qualify to ONNX proto specifications,
        the checker will throw errors, but in this case it successfully passes.
        This method confirms exported model protobuf is valid.
        Now, the model is ready to be imported in other frameworks for inference!
    '''

    model_proto = onnx.load(onnx_path)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)


# test시 nms 통과후 적용
def plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, reverse_rgb=False, absolute_coordinates=True,
              image_show=False, image_save=False, image_save_path=None, image_name=None):
    """Visualize bounding boxes.
    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.(range 0 ~ 255 - uint8)
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).
    """
    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if image_save:
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)

    img = img.astype(np.uint8)

    if len(bboxes) < 1:
        if isinstance(img, mx.nd.NDArray):
            img = img.asnumpy()
        if image_save:
            cv2.imwrite(os.path.join(image_save_path, image_name + ".jpg"), img)
        if image_show:
            cv2.imshow(image_name, img)
            cv2.waitKey(0)
    else:
        if isinstance(img, mx.nd.NDArray):
            img = img.asnumpy()
        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()
        if isinstance(labels, mx.nd.NDArray):
            labels = labels.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()

        if reverse_rgb:
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]

        copied_img = img.copy()

        if not absolute_coordinates:
            # convert to absolute coordinates using image shape
            height = img.shape[0]
            width = img.shape[1]
            bboxes[:, (0, 2)] *= width
            bboxes[:, (1, 3)] *= height

        # use random colors if None is provided
        if colors is None:
            colors = dict()

        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.ravel()[i] < thresh:  # threshold보다 작은 것 무시
                continue
            if labels is not None and labels.ravel()[i] < 0:  # 0이하 인것들 인것 무시
                continue

            cls_id = int(labels.ravel()[i]) if labels is not None else -1
            if cls_id not in colors:
                if class_names is not None and cls_id != -1:
                    colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
                else:
                    colors[cls_id] = (random.random(), random.random(), random.random())
            denorm_color = [x * 255 for x in colors[cls_id]]

            bbox[np.isinf(bbox)] = 0
            bbox[bbox < 0] = 0
            xmin, ymin, xmax, ymax = [int(np.rint(x)) for x in bbox]
            try:
                '''
                colors[cls_id] -> 기본적으로 list, tuple 자료형에 동작함

                numpy인 경우 float64만 동작함 - 나머지 동작안함
                다른 자료형 같은 경우는 tolist로 바꾼 다음 넣어줘야 동작함.
                '''
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), denorm_color, thickness=3)
            except Exception as E:
                print(E)

            if class_names is not None and cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = str(cls_id) if cls_id >= 0 else ''

            score = '{:.2f}'.format(scores.ravel()[i]) if scores is not None else ''

            if class_name or score:
                cv2.putText(copied_img,
                            text='{} {}'.format(class_name, score), \
                            org=(xmin + 7, ymin + 20), \
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, \
                            fontScale=0.5, \
                            color=[255, 255, 255], \
                            thickness=1, bottomLeftOrigin=False)

        result = cv2.addWeighted(img, 0.5, copied_img, 0.5, 0)

        if image_save:
            cv2.imwrite(os.path.join(image_save_path, image_name + ".jpg"), result)
        if image_show:
            cv2.imshow(image_name, result)
            cv2.waitKey(0)

        return result


"""Helper utils for export HybridBlock to symbols."""
# from gluoncv.utils import export_block 에서 아주 조금 수정
'''
c++에서의 일을 덜 수 있다.
Use preprocess=True will add a default preprocess layer above the network, 
which will subtract mean [123.675, 116.28, 103.53], divide std [58.395, 57.12, 57.375],
and convert original image (B, H, W, C and range [0, 255]) to tensor (B, C, H, W) as network input.
This is the default preprocess behavior of all GluonCV pre-trained models. With this preprocess head, 
you can use raw RGB image in C++ without explicitly applying these operations.
'''


class _DefaultPreprocess(HybridBlock):
    """Default preprocess block used by GluonCV.

    The default preprocess block includes:

        - mean [123.675, 116.28, 103.53]

        - std [58.395, 57.12, 57.375]

        - transpose to (B, 3, H, W)

    It is used to transform from resized original images with shape (1, H, W, 3) or (B, H, W, 3)
    in range (0, 255) and RGB color format.

    """

    def __init__(self, **kwargs):
        super(_DefaultPreprocess, self).__init__(**kwargs)
        with self.name_scope():
            mean = mx.nd.array([123.675, 116.28, 103.53]).reshape((1, 1, 1, 3))
            scale = mx.nd.array([58.395, 57.12, 57.375]).reshape((1, 1, 1, 3))
            self.init_mean = self.params.get_constant('init_mean', mean)
            self.init_scale = self.params.get_constant('init_scale', scale)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, init_mean, init_scale):
        x = F.broadcast_minus(x, init_mean)
        x = F.broadcast_div(x, init_scale)
        x = F.transpose(x, axes=(0, 3, 1, 2))
        return x


def export_block_for_cplusplus(path=None, block=None, data_shape=None, epoch=0, preprocess=True,
                               layout='HWC',
                               ctx=mx.cpu(), remove_amp_cast=True):
    """Helper function to export a HybridBlock to symbol JSON to be used by
    `SymbolBlock.imports`, `mxnet.mod.Module` or the C++ interface..

    Parameters
    ----------
    path : str
        Path to save model.
        Two files path-symbol.json and path-xxxx.params will be created,
        where xxxx is the 4 digits epoch number.
    block : mxnet.gluon.HybridBlock
        The hybridizable block. Note that normal gluon.Block is not supported.
    data_shape : tuple of int, default is None
        Fake data shape just for export purpose, in format (H, W, C).
        If you don't specify ``data_shape``, `export_block` will try use some common data_shapes,
        e.g., (224, 224, 3), (256, 256, 3), (299, 299, 3), (512, 512, 3)...
        If any of this ``data_shape`` goes through, the export will succeed.
    epoch : int
        Epoch number of saved model.
    preprocess : mxnet.gluon.HybridBlock, default is True.
        Preprocess block prior to the network.
        By default (True), it will subtract mean [123.675, 116.28, 103.53], divide
        std [58.395, 57.12, 57.375], and convert original image (B, H, W, C and range [0, 255]) to
        tensor (B, C, H, W) as network input. This is the default preprocess behavior of all GluonCV
        pre-trained models.
        You can use custom pre-process hybrid block or disable by set ``preprocess=None``.
    layout : str, default is 'HWC'
        The layout for raw input data. By default is HWC. Supports 'HWC' and 'CHW'.
        Note that image channel order is always RGB.
    ctx: mx.Context, default mx.cpu()
        Network context.

    Returns
    -------
    preprocessed block

    """
    # input image layout
    if data_shape is None:
        data_shapes = [(s, s, 3) for s in (224, 256, 299, 300, 320, 416, 512, 600)]
    else:
        data_shapes = [data_shape]

    '''
    c++ 에서 inference 할 때,
    데이터를 0 ~ 1 범위로 바꾸고, mxnet 입력형태인 (1, C, H, W)로 바꿀 필요가 없다.

    그대로 
    '''
    if preprocess:
        # add preprocess block
        if preprocess is True:
            preprocess = _DefaultPreprocess()
        else:
            if not isinstance(preprocess, HybridBlock):
                raise TypeError("preprocess must be HybridBlock, given {}".format(type(preprocess)))
        wrapper_block = HybridSequential()
        preprocess.initialize(ctx=ctx)
        wrapper_block.add(preprocess)
        wrapper_block.add(block)
    else:
        wrapper_block = block
    wrapper_block.collect_params().reset_ctx(ctx)

    # try different data_shape if possible, until one fits the network
    last_exception = None
    for dshape in data_shapes:
        h, w, c = dshape
        if layout == 'HWC':  # 보통 HWC(opencv)형식이다.
            x = mx.nd.zeros((1, h, w, c), ctx=ctx)
        elif layout == 'CHW':
            x = mx.nd.zeros((1, c, h, w), ctx=ctx)

        # hybridize and forward once
        wrapper_block.hybridize()
        try:
            wrapper_block(x)
            if path != None:
                wrapper_block.export(path, epoch, remove_amp_cast=remove_amp_cast)
            last_exception = None
            break
        except MXNetError as e:
            last_exception = e
    if last_exception is not None:
        raise RuntimeError(str(last_exception).splitlines()[0])
    return wrapper_block


# detector block + prediction block for mxnet c++
class PostNet(HybridBlock):

    def __init__(self, net=None, auxnet=None):
        super(PostNet, self).__init__()
        self._net = net
        self._auxnet = auxnet

    def hybrid_forward(self, F, x):
        output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = self._net(
            x)
        '''
            hybrid block은 symbol 아니면 ndarray를 받아야한다.
            네트워크가 출력하는 형태를 알고 있으니 다음과 같이 직접 쪼개서 보내자
        '''
        return self._auxnet(output1, output2, output3,
                            anchor1, anchor2, anchor3,
                            offset1, offset2, offset3,
                            stride1, stride2, stride3)


class YoloV3output(HybridBlock):

    def __init__(self, Darknetlayer=53,
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=1,  # foreground만
                 pretrained=True,
                 pretrained_path="modelparam",
                 ctx=mx.cpu()):
        super(YoloV3output, self).__init__()

        if Darknetlayer not in [53]:
            raise ValueError

        anchors = OrderedDict(anchors)
        anchors = list(anchors.values())[::-1]

        # 각 레이어에서 나오는 anchor갯수가 바뀔수도 있으니!!
        self._num_anchors = []
        for anchor in anchors:
            self._num_anchors.append(len(anchor))  # 변화 가능

        self._darkenet = get_darknet(Darknetlayer, pretrained=pretrained, ctx=ctx, root=pretrained_path)
        self._num_classes = num_classes
        self._num_pred = 5 + num_classes  # 고정

        with self.name_scope():

            head_init_num_channel = 512
            trans_init_num_channel = 256
            self._head = HybridSequential()
            self._transition = HybridSequential()
            self._upsampleconv = HybridSequential()

            # output
            for j in range(len(anchors)):
                if j == 0:
                    factor = 1
                else:
                    factor = 2
                head_init_num_channel = head_init_num_channel // factor
                for _ in range(3):
                    self._head.add(Conv2D(channels=head_init_num_channel,
                                          kernel_size=(1, 1),
                                          strides=(1, 1),
                                          padding=(0, 0),
                                          use_bias=True,
                                          in_channels=0,
                                          weight_initializer=mx.init.Normal(0.01),
                                          bias_initializer='zeros'
                                          ))
                    self._head.add(BatchNorm(epsilon=1e-5, momentum=0.9))
                    self._head.add(LeakyReLU(0.1))

                    self._head.add(Conv2D(channels=head_init_num_channel * 2,
                                          kernel_size=(3, 3),
                                          strides=(1, 1),
                                          padding=(1, 1),
                                          use_bias=True,
                                          in_channels=0,
                                          weight_initializer=mx.init.Normal(0.01),
                                          bias_initializer='zeros'
                                          ))
                    self._head.add(BatchNorm(epsilon=1e-5, momentum=0.9))
                    self._head.add(LeakyReLU(0.1))

                self._head.add(Conv2D(channels=len(anchors[j]) * self._num_pred,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding=(0, 0),
                                      use_bias=True,
                                      in_channels=0,
                                      weight_initializer=mx.init.Normal(0.01),
                                      bias_initializer='zeros'
                                      ))

            # for upsample - transition
            for i in range(len(anchors) - 1):
                if i == 0:
                    factor = 1
                else:
                    factor = 2
                trans_init_num_channel = trans_init_num_channel // factor
                self._transition.add(Conv2D(channels=trans_init_num_channel,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding=(0, 0),
                                            use_bias=True,
                                            in_channels=0,
                                            weight_initializer=mx.init.Normal(0.01),
                                            bias_initializer='zeros'
                                            ))
                self._transition.add(BatchNorm(epsilon=1e-5, momentum=0.9))
                self._transition.add(LeakyReLU(0.1))

            # for deconvolution upsampleing
            for i in range(len(anchors) - 1):
                if i == 0:
                    factor = 1
                else:
                    factor = 2
                trans_init_num_channel = trans_init_num_channel // factor
                self._upsampleconv.add(Conv2DTranspose(trans_init_num_channel, kernel_size=3, strides=2, padding=1,
                                                       output_padding=1, use_bias=True, in_channels=0))
                self._upsampleconv.add(BatchNorm(epsilon=1e-5, momentum=0.9))
                self._upsampleconv.add(LeakyReLU(0.1))

        self._head.initialize(ctx=ctx)
        self._transition.initialize(ctx=ctx)
        self._upsampleconv.initialize(ctx=ctx)
        print(f"{self.__class__.__name__} Head weight init 완료")

    def hybrid_forward(self, F, x):

        feature_36, feature_61, feature_74 = self._darkenet(x)

        # first
        transition = self._head[:15](feature_74)  # darknet 기준 75 ~ 79
        output82 = self._head[15:19](transition)  # darknet 기준 79 ~ 82

        # second
        transition = self._transition[0:3](transition)

        # transition = F.UpSampling(transition, scale=2,
        #                           sample_type='nearest')  # or sample_type = "bilinear" , 후에 deconvolution으로 대체
        transition = self._upsampleconv[0:3](transition)

        transition = F.concat(transition, feature_61, dim=1)

        transition = self._head[19:34](transition)  # darknet 기준 75 ~ 91
        output94 = self._head[34:38](transition)  # darknet 기준 91 ~ 82

        # third
        transition = self._transition[3:](transition)

        # transition = F.UpSampling(transition, scale=2,
        #                           sample_type='nearest')  # or sample_type = "bilinear" , 후에 deconvolution으로 대체
        transition = self._upsampleconv[3:](transition)

        transition = F.concat(transition, feature_36, dim=1)
        output106 = self._head[38:](transition)  # darknet 기준 91 ~ 106

        output82 = F.transpose(output82,
                               axes=(0, 2, 3, 1))
        output94 = F.transpose(output94, axes=(0, 2, 3, 1))
        output106 = F.transpose(output106, axes=(0, 2, 3, 1))

        # (batch size, height, width, len(anchors), (5 + num_classes)
        return output82, output94, output106


class YoloAnchorGenerator(HybridBlock):

    def __init__(self, index, anchor, feature, stride):
        super(YoloAnchorGenerator, self).__init__()
        width, height = feature
        anchor = np.reshape(anchor, (1, 1, -1, 2))
        # grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        grid_y, grid_x = np.mgrid[:height, :width]
        offset = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)  # (13, 13, 2)
        offset = np.expand_dims(offset, axis=0)  # (1, 13, 13, 2)
        offset = np.expand_dims(offset, axis=3)  # (1, 13, 13, 1, 2)
        stride = np.reshape(stride, (1, 1, 1, 2))

        with self.name_scope():
            self.anchor = self.params.get_constant(f'anchor_{index}', value=anchor)
            self.offset = self.params.get_constant(f'offset_{index}', value=offset)
            self.stride = self.params.get_constant(f'stride_{index}', value=stride)

    def hybrid_forward(self, F, x, anchor, offset, stride):
        '''
            mxnet 형식의 weight로 추출 할때는 아래의 과정이 필요 없으나, onnx weight로 추출 할 때 아래와 같이
            F.identity로 감싸줘야, onnx_mxnet.export_model 에서 anchors, offsets, strides 를 출력으로 인식한다.
            (anchor, offset, stride = self._anchor_generators[i](x) 을 바로 출력하면 안되고, F.identity로 감싸줘야
            mxnet의 연산으로 인식하는 듯.)
            mxnet 의 onnx 관련 API들이 완벽하진 않은듯.
            또한 hybridize를 사용하려면 아래와 같이 3개로 감싸줘야 한다. 아래의 F.identity를 사용하지 않고 hybridize를 한다면,
            mxnet.base.MXNetError: Error in operator node_0_backward:
            [22:45:32] c:\jenkins\workspace\mxnet-tag\mxnet\src\imperative\./imperative_utils.h:725: Check failed:
            g.GetAttr<size_t>("storage_type_num_unknown_nodes") == 0U (9 vs. 0) :
            위와 같은 에러가 발생한다.
        '''

        anchor = F.identity(anchor)
        offset = F.identity(offset)
        stride = F.identity(stride)
        return anchor, offset, stride


class AnchorOffstNet(HybridBlock):

    def __init__(self, net=None, version=None, anchors=None, target_size=None,
                 ctx=mx.cpu()):
        super(AnchorOffstNet, self).__init__()
        self._net = net

        features = []
        strides = []
        darknet_output = get_darknet(version, pretrained=False, ctx=mx.cpu(), dummy=True)(
            mx.nd.random_uniform(low=0, high=1, shape=(1, 3, target_size[0], target_size[1]), ctx=mx.cpu()))
        for out in darknet_output:  # feature_14, feature_24, feature_28
            out_height, out_width = out.shape[2:]
            features.append([out_width, out_height])
            strides.append([target_size[1] // out_width, target_size[0] // out_height])

        features = features[::-1]
        strides = strides[::-1]  # deep -> middle -> shallow 순으로 !!!
        anchors = OrderedDict(anchors)
        anchors = list(anchors.values())[::-1]
        self._numoffst = len(anchors)

        with self.name_scope():
            self._anchor_generators = HybridSequential()
            for i, anchor, feature, stride in zip(range(len(features)), anchors, features, strides):
                self._anchor_generators.add(YoloAnchorGenerator(i, anchor, feature, stride))

        self._anchor_generators.initialize(ctx=ctx)

    def hybrid_forward(self, F, x):

        output82, output94, output106 = self._net(x)

        anchors = []
        offsets = []
        strides = []
        for i in range(self._numoffst):
            anchor, offset, stride = self._anchor_generators[i](x)
            anchors.append(anchor)
            offsets.append(offset)
            strides.append(stride)

        return output82, output94, output106, \
               anchors[0], anchors[1], anchors[2], \
               offsets[0], offsets[1], offsets[2], \
               strides[0], strides[1], strides[2]
