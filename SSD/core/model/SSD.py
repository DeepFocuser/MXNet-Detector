# gluoncv에 있는 코드 참고

import logging
import os

import mxnet as mx
import numpy as np
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential, Conv2D

from core.model.backbone.VGG16 import VGG16

'''
< Network >
VGG16를 기반으로 하고
fc6, fc7 -> convolution layer로 대체
pool5(2x2, stride 2) -> pool5(3x3. stride 1)
atrous algorithm사용 
dropout, fc8 layer 제거
optimizer : SGD, Learning rate : 0.001, momentum : 0.9, weight decay : 0.0005, batch size : 32
The learning rate decay policy is slightly different for each dataset(이건 뭐...)

input_size = 300 x 300 일때,
< 
  conv4_3 : 4 default box 
  conv7 : 6 default box
  conv8_2 : 6 default box  
  conv9_2 : 6 default box
  conv10_2 : 4 default box
  conv11_2 : 4 default box
>

input_size = 512 x 512 일때,
< 
  conv4_3 : 4 default box 
  conv7 : 6 default box
  conv8_2 : 6 default box  
  conv9_2 : 6 default box
  conv10_2 : 6 default box
  conv11_2 : 4 default box
  conv12_2 : 4 default box
>

사용(4 default box에 대해서는 aspect ratios 1/3, 3 생략)

conv4_3은 다른 feature scale을 가지고 있기 때문에, L2 Normalization 적용 
        
< scale > 
voc300 : (conv4_3 : 0.1), Smin : 0.2   
--> 0.1, 0.2, 0.375, 0.55, 0,725, 0,9, 1.075
--> 30, 60, 112.5, 165, 217.5, 270, 323

voc512 : (conv4_3 : 0.07), Smin : 0.15
+ conv12_2 
--> 0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05
--> 35.84, 76.8, 153.6, 230.4, 307.2, 384, 460.8, 537.6

coco300 : (conv4_3 : 0.07), Smin : 0.15
--> 0.07, 0.15, 0.3375, 0.525, 0.7125, 0.9, 1.0875
--> 21, 45, 101.25, 157.5, 213.75, 270, 326.25

------------------------------------------------------------
coco512 : (conv4_3 : 0.04) , Smin : 0.1
+ conv12_2 
--> 0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06
--> 21, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72
------------------------------------------------------------

we initialize the parameters for all the newly added convolution layer with 
the 'XAVIER' method

Sk = Smin + (Smax(0.9) - Smin / m-1)*(K-1), K is between [1, m] , m is feature maps
--> 주의 할 것 conv4_3은 보통 scale이 주어져 있으므로, 실제 m = m-1 이다 
--> 마지막 feature를 위해서 m+1로 scale하나 더 구해줘야함. ->  323, 537.6, 326.25, 542.72과 같이..
'''


class SSDAnchorGenerator(HybridBlock):

    def __init__(self, index=None, feature_size=None, input_size=None, box_size=None, box_ratio=None,
                 box_offset=(0.5, 0.5), box_clip=False, alloc_size=[256, 256]):
        super(SSDAnchorGenerator, self).__init__()
        self._height = input_size[0]
        self._width = input_size[1]
        self._box_clip = box_clip

        # 아래와 같이 등록해주면 hybrid_forward에 anchors로서 들어감
        with self.name_scope():
            self.anchors = self.params.get_constant(name=f'anchor_{index}',
                                                    value=self._generate_anchors(feature_size=feature_size,
                                                                                 box_size=(
                                                                                     box_size[0], np.sqrt(
                                                                                         box_size[0] * box_size[1])),
                                                                                 box_ratio=box_ratio,
                                                                                 box_offset=box_offset,
                                                                                 alloc_size=alloc_size))

    def _generate_anchors(self, feature_size=None, box_size=None, box_ratio=None, box_offset=None, alloc_size=None):
        # cx, cy, w, h 순으로 붙임!!!

        y_range = max(feature_size[0], alloc_size[0])
        x_range = max(feature_size[1], alloc_size[1])

        step_y = self._height / feature_size[0]
        step_x = self._width / feature_size[1]

        anchors = []
        for y in range(y_range):
            for x in range(x_range):

                cy = (y + box_offset[0]) * step_y
                cx = (x + box_offset[1]) * step_x

                anchors.append([cx, cy, box_size[0], box_size[0]])
                anchors.append([cx, cy, box_size[1], box_size[1]])
                for r in box_ratio[1:]:
                    sr = np.sqrt(r)
                    w = box_size[0] * sr
                    h = box_size[0] / sr
                    anchors.append([cx, cy, w, h])
        anchors = np.array(anchors)

        if self._box_clip:
            cx, cy, w, h = np.split(anchors, 4, axis=-1)
            anchors = np.concatenate([np.clip(cx, 0, self._width),
                                      np.clip(cy, 0, self._height),
                                      np.clip(w, 0, self._width),
                                      np.clip(h, 0, self._height),
                                      ], axis=-1)
        return anchors.reshape(1, 1, y_range, x_range, -1)

    def hybrid_forward(self, F, x, anchors):
        x = F.expand_dims(x, axis=-1)
        anchors = F.slice_like(anchors, x, axes=(2, 3))
        return F.reshape(anchors, shape=(-1, 4))


class ConvPredictor(HybridBlock):

    def __init__(self, num_channel=None,
                 kernel=(3, 3),
                 pad=(1, 1),
                 stride=(1, 1),
                 activation=None,
                 use_bias=True,
                 in_channels=0,
                 weight_initializer=mx.init.Xavier(rnd_type="uniform", factor_type="avg", magnitude=3)):
        super(ConvPredictor, self).__init__()
        with self.name_scope():
            self.predictor = Conv2D(
                num_channel, kernel, strides=stride, padding=pad,
                activation=activation, use_bias=use_bias, in_channels=in_channels,
                weight_initializer=weight_initializer,
                bias_initializer='zeros')

    def hybrid_forward(self, F, x):
        return self.predictor(x)


class SSD_VGG16(HybridBlock):

    def __init__(self, version=512, input_size=(512, 512),
                 box_sizes=[21, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
                 box_ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0 / 3]] * 4 + [[1, 2, 0.5]] * 2,
                 num_classes=1,
                 pretrained=False,
                 pretrained_path="modelparam",
                 anchor_box_offset=(0.5, 0.5),
                 anchor_box_clip=False,
                 alloc_size=[256, 256],
                 ctx=mx.cpu()):
        super(SSD_VGG16, self).__init__()

        if version not in [300, 512]:
            raise ValueError

        if len(box_sizes) - 1 != len(box_ratios):
            raise ValueError

        feature_sizes = []
        fetures_output = VGG16(version=version, ctx=mx.cpu(), dummy=True)(
            mx.nd.random_uniform(low=0, high=1, shape=(1, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
        for fpn in fetures_output:
            feature_sizes.append(fpn.shape[2:])  # h, w

        self._features = VGG16(version=version, pretrained=pretrained, ctx=ctx, root=pretrained_path)
        self._num_classes = num_classes
        sizes = list(zip(box_sizes[:-1], box_sizes[1:]))

        with self.name_scope():

            self._class_predictors = HybridSequential()
            self._box_predictors = HybridSequential()
            self._anchor_generators = HybridSequential()
            for index, size, ratio, feature_size in zip(range(len(feature_sizes)), sizes, box_ratios, feature_sizes):
                self._class_predictors.add(ConvPredictor(num_channel=(len(ratio) + 1) * (num_classes + 1),
                                                         kernel=(3, 3),
                                                         pad=(1, 1),
                                                         stride=(1, 1),
                                                         activation=None,
                                                         use_bias=True,
                                                         in_channels=0,
                                                         weight_initializer=mx.init.Xavier(rnd_type="uniform",
                                                                                           factor_type="avg",
                                                                                           magnitude=3)))

                # activation = None 인 것 주의
                self._box_predictors.add(ConvPredictor(num_channel=(len(ratio) + 1) * 4,
                                                       kernel=(3, 3),
                                                       pad=(1, 1),
                                                       stride=(1, 1),
                                                       activation=None,
                                                       use_bias=True,
                                                       in_channels=0,
                                                       weight_initializer=mx.init.Xavier(rnd_type="uniform",
                                                                                         factor_type="avg",
                                                                                         magnitude=3)))
                self._anchor_generators.add(SSDAnchorGenerator(index=index,
                                                               feature_size=feature_size,
                                                               input_size=input_size,
                                                               box_size=size,
                                                               box_ratio=ratio,
                                                               box_offset=anchor_box_offset,
                                                               box_clip=anchor_box_clip,
                                                               alloc_size=(alloc_size[0] // (2 ** index),
                                                                           alloc_size[1] // (2 ** index))))

        self._class_predictors.initialize(ctx=ctx)
        self._box_predictors.initialize(ctx=ctx)
        self._anchor_generators.initialize(ctx=ctx)
        logging.info(f"{self.__class__.__name__} Head weight init 완료")

    def hybrid_forward(self, F, x):
        # 1. VGG16 Feature
        feature_list = self._features(x)

        # 2. class, box prediction
        cls_preds = [F.flatten(data=F.transpose(data=class_prediction(feature), axes=(0, 2, 3, 1)))
                     # (batch, height, width, class)
                     for feature, class_prediction in zip(feature_list, self._class_predictors)]
        box_preds = [F.flatten(data=F.transpose(data=box_predictor(feature), axes=(0, 2, 3, 1)))
                     # (batch, height, width, box)
                     for feature, box_predictor in zip(feature_list, self._box_predictors)]
        # feature가 anchor_generator에 통과는 하는데 사용하지 않음.
        anchors = [anchor_generator(feature) for feature, anchor_generator in
                   zip(feature_list, self._anchor_generators)]
        ''' 
        shape=(0,..)에서 0은 무엇인가?
        mxnet reshape의 특징인데,
        자세한 설명은 https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#mxnet.ndarray.NDArray.reshape 에 있고,
        간략히 설명하자면, 0 copy this dimension from the input to the output shape 이라고 한다.
        -1, -2, -3, -4 에 대한 설명도 있으니, 나중에 사용하게 되면 참고하자.
        첫번째 축을 명시해 주지 않아도 되는 장점이 있다.
        '''
        # https://github.com/apache/incubator-mxnet/issues/13998 - expand_dims은 copy한다.
        # expand_dims() makes copy instead of simply reshaping - 아래와 같은 경우 reshape을 사용하자.
        cls_preds = F.reshape(data=F.concat(*cls_preds, dim=-1), shape=(0, -1, self._num_classes + 1))
        box_preds = F.reshape(data=F.concat(*box_preds, dim=-1), shape=(0, -1, 4))

        # anchors = F.concat(*anchors, dim=0).expand_dims(axis=0) #
        anchors = F.reshape(F.concat(*anchors, dim=0), shape=(1, -1, 4))
        # anchors = F.concat(*anchors, dim=0)
        return cls_preds, box_preds, anchors


if __name__ == "__main__":
    input_size = (512, 512)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    net = SSD_VGG16(version=512, input_size=input_size,
                    # box_sizes=[21, 45, 101.25, 157.5, 213.75, 270, 326.25],
                    box_sizes=[21, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
                    # box_ratios=[[1, 2, 0.5]] +  # conv4_3
                    #           [[1, 2, 0.5, 3, 1.0 / 3]] * 3 +  # conv7, conv8_2, conv9_2
                    #           [[1, 2, 0.5]] * 2,  # conv10_2, conv11_2
                    box_ratios=[[1, 2, 0.5]] +  # conv4_3
                               [[1, 2, 0.5, 3, 1.0 / 3]] * 4 +  # conv7, conv8_2, conv9_2, conv10_2
                               [[1, 2, 0.5]] * 2,  # conv11_2, conv12_2
                    num_classes=5,
                    pretrained=False,
                    pretrained_path=os.path.join(root, "modelparam"),
                    anchor_box_offset=(0.5, 0.5),
                    anchor_box_clip=True,
                    alloc_size=[256, 256],
                    ctx=mx.cpu())

    net.hybridize(active=True, static_alloc=True, static_shape=True)

    cls_preds, box_preds, anchors = net(
        mx.nd.random_uniform(low=0, high=1, shape=(8, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
    print(f"< input size(height, width) : {input_size} >")
    print(f"< class prediction shape : {cls_preds.shape} >")
    print(f"< box prediction shape : {box_preds.shape} >")
    print(f"< anchor shape : {anchors.shape} >")
    '''
    < input size(height, width) : (512, 512) >
    < class prediction shape : (8, 24564, 6) >
    < box prediction shape : (8, 24564, 4) >
    < anchor shape : (1, 393024, 4) >
    '''
