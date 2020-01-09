# https://github.com/mnikitin/EfficientNet 참고

import logging
import os
from math import ceil

import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential, Conv2D, BatchNorm, Activation

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def ConvBatchAct(channels=1, kernel=1, stride=1, pad=0, num_group=1, active=True):
    net = HybridSequential()
    with net.name_scope():
        net.add(Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
        net.add(BatchNorm(momentum=0.9))
        if active:
            net.add(Activation('relu'))
            # net.add(Swish())
    return net


# MnasNet: Platform-Aware Neural Architecture Search for Mobile - https://arxiv.org/pdf/1807.11626.pdf
class MBConv(HybridBlock):

    def __init__(self, in_channels, channels, t, kernel, stride, **kwargs):
        super(MBConv, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        self.net = HybridSequential()

        with self.net.name_scope():
            self.net.add(ConvBatchAct(in_channels * t))
            self.net.add(ConvBatchAct(in_channels * t,
                                      kernel=kernel,
                                      stride=stride,
                                      pad=int((kernel - 1) / 2),
                                      num_group=in_channels * t))
            self.net.add(ConvBatchAct(channels, active=True))

    def hybrid_forward(self, F, x):
        out = self.net(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class EfficientBase(HybridBlock):
    '''
    depth(alpha)는 올림
    width(beta) 는 내림
    '''

    def __init__(self, alpha=1.0, beta=1.0, dropout_rate=0.2, classes=1000, **kwargs):

        super(EfficientBase, self).__init__(**kwargs)

        # base model 0 settings
        repeats = [1, 2, 2, 3, 3, 4, 1]
        channels_num = [16, 24, 40, 80, 112, 192, 320]
        kernels_num = [3, 3, 5, 3, 5, 5, 3]
        t_num = [1, 6, 6, 6, 6, 6, 6]  # MBConv 인자
        strides_first = [1, 2, 2, 2, 1, 2, 1]

        self._repeats = repeats
        self._alpha = alpha
        self.features = HybridSequential()
        with self.features.name_scope():
            self.features.add(ConvBatchAct(channels=int(32 * beta), kernel=3, stride=2, pad=1, active=True))

            # params of MBConv layers
            in_channels_group = []
            for rep, ch_num in zip([1] + repeats[:-1], [32] + channels_num[:-1]):
                in_channels_group += [int(ch_num * beta)] * ceil(alpha * rep)

            channels_group, kernels, ts, strides = [], [], [], []
            for rep, ch, kernel, t, s in zip(repeats, channels_num, kernels_num, t_num, strides_first):
                rep = ceil(alpha * rep)
                channels_group += [int(ch * beta)] * rep
                kernels += [kernel] * rep
                ts += [t] * rep
                strides += [s] + [1] * (rep - 1)

            # add MBConv layers
            for in_c, c, t, k, s in zip(in_channels_group, channels_group, ts, kernels, strides):
                self.features.add(MBConv(in_channels=in_c, channels=c, t=t, kernel=k, stride=s))

            # p6, p7은 내 마음대로 함.
            self.features.add(
                MBConv(in_channels=int(channels_num[-1] * beta), channels=int(channels_num[-1] * beta), t=t_num[-1], kernel=3, stride=2))
            self.features.add(
                MBConv(in_channels=int(channels_num[-1] * beta), channels=int(channels_num[-1] * beta), t=t_num[-1], kernel=3, stride=2))

            # for classification
            # head layers
            # last_channels = int(1280 * beta) if beta > 1.0 else 1280
            # _add_conv(self.features, last_channels)
            # self.features.add(GlobalAvgPool2D())
            #
            # # features dropout
            # self.dropout = Dropout(dropout_rate) if dropout_rate > 0.0 else None
            # self.output = HybridSequential()
            # with self.output.name_scope():
            #     self.output.add(Dense(classes, use_bias=True, flatten=True))

    def hybrid_forward(self, F, x):

        # repeats = [1, 2, 2, 3, 3, 4, 1]
        p3_index = ceil(self._alpha * self._repeats[0]) \
                   + ceil(self._alpha * self._repeats[1]) \
                   + ceil(self._alpha * self._repeats[2]) + 1

        p4_index = p3_index + ceil(self._alpha * self._repeats[3])
        p5_index = p4_index + ceil(self._alpha * self._repeats[4]) + ceil(self._alpha * self._repeats[5])
        p6_index = p5_index + ceil(self._alpha * self._repeats[-1]) + 1

        # version에 따라서 쪼개야함
        p3 = self.features[0:p3_index](x)  # input / 8
        p4 = self.features[p3_index:p4_index](p3)  # input / 16
        p5 = self.features[p4_index:p5_index](p4)  # input / 32
        p6 = self.features[p5_index:p6_index](p5)  # input / 64
        p7 = self.features[p6_index:](p6)  # input / 128
        # if self.dropout:
        #     x = self.dropout(x)
        # x = self.output(x)
        return p3, p4, p5, p6, p7


def get_efficientbase(version=None):
    ''' 입력 사이즈
    B0 -> 512 x 512
    B1 -> 640 x 640
    B2 -> 768 x 768
    ------------------- 여기까지가 실시간으로 쓸만 할듯 -------------------
    B3 -> 896 x 896
    B4 -> 1024 x 1024
    B5 -> 1280 x 1280
    B6 -> 1408 x 1408
    B7 -> 1536 x 1536
    '''

    params_dict = \
        {
            # (depth(layer), width(channel), dropout_rate)
            0: (1.0, 1.0, 0.2),
            1: (pow(1.2, 2), pow(1.1, 2), 0.242),
            2: (pow(1.2, 3), pow(1.1, 3), 0.285),
            3: (pow(1.2, 4), pow(1.1, 4), 0.328),
            4: (pow(1.2, 5), pow(1.1, 5), 0.371),
            5: (pow(1.2, 6), pow(1.1, 6), 0.414),
            6: (pow(1.2, 7), pow(1.1, 7), 0.457),
            7: (pow(1.2, 8), pow(1.1, 8), 0.5)
        }
    depth, width, dropout_rate = params_dict[version]
    net = EfficientBase(alpha=depth, beta=width, dropout_rate=dropout_rate)
    return net


if __name__ == "__main__":
    input_size = (512, 512)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    net = get_efficientbase(0)
    net.initialize(ctx=mx.cpu())
    net.hybridize(active=True, static_alloc=True, static_shape=True)
    output = net(mx.nd.random_uniform(low=0, high=1, shape=(2, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
    print(f"< input size(height, width) : {input_size} >")
    print(f"< output number : {len(output)} >")
    for i, out in enumerate(output):
        print(f"({i + 1}) feature shape : {out.shape}")
    '''
        < input size(height, width) : (512, 512) >
        < output number : 5 >
        (1) feature shape : (2, 40, 64, 64)
        (2) feature shape : (2, 80, 32, 32)
        (3) feature shape : (2, 192, 16, 16)
        (4) feature shape : (2, 320, 8, 8)
        (5) feature shape : (2, 320, 4, 4)
    '''
