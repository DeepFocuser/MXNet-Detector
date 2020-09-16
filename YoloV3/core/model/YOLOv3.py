from collections import OrderedDict

import numpy as np

from core.model.backbone.Darknet import *


class YoloAnchorGenerator(HybridBlock):

    def __init__(self, index, anchor, feature, stride, alloc_size):
        super(YoloAnchorGenerator, self).__init__()

        fwidth, fheight = feature
        aheight, awidth = alloc_size
        width = max(fwidth, awidth)
        height = max(fheight, aheight)

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


class Yolov3(HybridBlock):

    def __init__(self, Darknetlayer=53,
                 input_size=(416, 416),
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=1,  # foreground만
                 pretrained=True,
                 pretrained_path="modelparam",
                 alloc_size=(64, 64),
                 ctx=mx.cpu()):
        super(Yolov3, self).__init__()

        if Darknetlayer not in [53]:
            raise ValueError

        in_height, in_width = input_size
        features = []
        strides = []
        anchors = OrderedDict(anchors)
        anchors = list(anchors.values())[::-1]
        self._numoffst = len(anchors)

        darknet_output = get_darknet(Darknetlayer, ctx=mx.cpu(), dummy=True)(
            mx.nd.random_uniform(low=0, high=1, shape=(1, 3, in_height, in_width), ctx=mx.cpu()))
        for out in darknet_output:  # feature_14, feature_24, feature_28
            out_height, out_width = out.shape[2:]
            features.append([out_width, out_height])
            strides.append([in_width // out_width, in_height // out_height])  # w, h

        features = features[::-1]
        strides = strides[::-1]  # deep -> middle -> shallow 순으로 !!!
        self._darkenet = get_darknet(Darknetlayer, pretrained=pretrained, ctx=ctx, root=pretrained_path)
        self._num_classes = num_classes
        self._num_pred = 5 + num_classes  # 고정

        with self.name_scope():

            head_init_num_channel = 512
            trans_init_num_channel = 256
            self._head = HybridSequential()
            self._transition = HybridSequential()
            self._upsampleconv = HybridSequential()
            self._anchor_generators = HybridSequential()

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

            # trans_init_num_channel = 256
            # # for deconvolution upsampleing
            # for i in range(len(anchors) - 1):
            #     if i == 0:
            #         factor = 1
            #     else:
            #         factor = 2
            #     trans_init_num_channel = trans_init_num_channel // factor
            #     self._upsampleconv.add(Conv2DTranspose(trans_init_num_channel, kernel_size=3, strides=2, padding=1,
            #                                            output_padding=1, use_bias=True, in_channels=0))
            #     self._upsampleconv.add(BatchNorm(epsilon=1e-5, momentum=0.9))
            #     self._upsampleconv.add(LeakyReLU(0.1))

        for i, anchor, feature, stride in zip(range(len(anchors)), anchors, features, strides):
            self._anchor_generators.add(
                YoloAnchorGenerator(i, anchor, feature, stride, (alloc_size[0] * (2 ** i), alloc_size[1] * (2 ** i))))

        self._head.initialize(ctx=ctx)
        self._transition.initialize(ctx=ctx)
        self._upsampleconv.initialize(ctx=ctx)
        self._anchor_generators.initialize(ctx=ctx)
        logging.info(f"{self.__class__.__name__} Head weight init 완료")

    def hybrid_forward(self, F, x):

        feature_36, feature_61, feature_74 = self._darkenet(x)
        # first
        transition = self._head[:15](feature_74)  # darknet 기준 75 ~ 79
        output82 = self._head[15:19](transition)  # darknet 기준 79 ~ 82

        # second
        transition = self._transition[0:3](transition)

        transition = F.UpSampling(transition, scale=2,
                                  sample_type='nearest')  # or sample_type = "bilinear" , 후에 deconvolution으로 대체
        # transition = self._upsampleconv[0:3](transition)

        transition = F.concat(transition, feature_61, dim=1)
        transition = self._head[19:34](transition)  # darknet 기준 75 ~ 91
        output94 = self._head[34:38](transition)  # darknet 기준 91 ~ 82

        # third
        transition = self._transition[3:](transition)

        transition = F.UpSampling(transition, scale=2,
                                  sample_type='nearest')  # or sample_type = "bilinear" , 후에 deconvolution으로 대체
        # transition = self._upsampleconv[3:](transition)

        transition = F.concat(transition, feature_36, dim=1)

        output106 = self._head[38:](transition)  # darknet 기준 91 ~ 106

        output82 = F.transpose(output82,
                               axes=(0, 2, 3, 1))
        output94 = F.transpose(output94, axes=(0, 2, 3, 1))
        output106 = F.transpose(output106, axes=(0, 2, 3, 1))

        # (batch size, height, width, len(anchors), (5 + num_classes)
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
               strides[0], strides[1], strides[2],


if __name__ == "__main__":

    input_size = (608, 608)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    net = Yolov3(Darknetlayer=53,
                 input_size=input_size,
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=5,  # foreground만
                 pretrained=False,
                 pretrained_path=os.path.join(root, "modelparam"),
                 alloc_size=(64, 64),
                 ctx=mx.gpu(0))
    net.hybridize(active=True, static_alloc=True, static_shape=True)
    output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(
        mx.nd.random_uniform(low=0, high=1, shape=(16, 3, input_size[0], input_size[1]), ctx=mx.gpu(0)))

    print(f"< input size(height, width) : {input_size} >")
    for i, pred in enumerate([output1, output2, output3]):
        print(f"prediction {i + 1} : {pred.shape}")
    for i, anchor in enumerate([anchor1, anchor2, anchor3]):
        print(f"anchor {i + 1} w, h 순서 : {anchor.shape}")
    for i, offset in enumerate([offset1, offset2, offset3]):
        print(f"offset {i + 1} w, h 순서 : {offset.shape}")
    for i, stride in enumerate([stride1, stride2, stride3]):
        print(f"stride {i + 1} w, h 순서 : {stride.shape}")
    '''
    < input size(height, width) : (608, 608) >
    prediction 1 : (1, 19, 19, 30)
    prediction 2 : (1, 38, 38, 30)
    prediction 3 : (1, 76, 76, 30)
    anchor 1 w, h 순서 : (1, 1, 3, 2)
    anchor 2 w, h 순서 : (1, 1, 3, 2)
    anchor 3 w, h 순서 : (1, 1, 3, 2)
    offset 1 w, h 순서 : (1, 64, 64, 1, 2)
    offset 2 w, h 순서 : (1, 128, 128, 1, 2)
    offset 3 w, h 순서 : (1, 256, 256, 1, 2)
    stride 1 w, h 순서 : (1, 1, 1, 2)
    stride 2 w, h 순서 : (1, 1, 1, 2)
    stride 3 w, h 순서 : (1, 1, 1, 2)
    '''
