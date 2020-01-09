import logging
import os

import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential, Conv2D, MaxPool2D, BatchNorm, Activation

from core.model.backbone.EfficientBase import get_efficientbase

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def lateral_conv(channels, kernel_size, strides, padding):
    lateral = HybridSequential()
    with lateral.name_scope():
        lateral.add(Conv2D(channels, kernel_size, strides, padding))
        lateral.add(BatchNorm(momentum=0.9, epsilon=1e-5))
        lateral.add(Activation('relu'))
    return lateral


class intermediateFF(HybridBlock):

    def __init__(self, channels, momentum=0.9, epsilon=1e-5):
        super(intermediateFF, self).__init__()

        with self.name_scope():
            # self.weight1 = self.params.get('weight1', shape=(1,))
            # self.weight2 = self.params.get('weight2', shape=(1,))

            # depthwise separable convolution
            self._dws = HybridSequential()
            self._dws.add(Conv2D(channels, 3, strides=(1, 1), padding=(1, 1),
                                 groups=channels))
            self._dws.add(BatchNorm(momentum=momentum, epsilon=epsilon))
            self._dws.add(Activation('relu'))
            self._dws.add(Conv2D(channels, 1, strides=(1, 1), padding=(0, 0),
                                 groups=1))
            self._dws.add(BatchNorm(momentum=momentum, epsilon=epsilon))
            self._dws.add(Activation('relu'))

    def hybrid_forward(self, F, x1, x2, weight1=None, weight2=None, epsilon=0.0001):
        '''
        In paper, weight* is ensured by applying a RELU activation Function - 이거 학습 안됨.
        '''
        # w1 = F.relu(weight1)
        # w2 = F.relu(weight2)
        #
        # total = w1 + w2 + epsilon
        # w1 = F.broadcast_div(w1, total)
        # w2 = F.broadcast_div(w2, total)
        #
        # first = F.broadcast_mul(w1, x1)
        # second = F.broadcast_mul(w2, x2)

        first = x1
        second = x2
        x = first + second
        x = self._dws(x)
        return x


class OutputFF(HybridBlock):

    def __init__(self, channels, momentum=0.9, epsilon=1e-5):
        super(OutputFF, self).__init__()
        with self.name_scope():
            # self.weight1 = self.params.get('weight1', shape=(1,))
            # self.weight2 = self.params.get('weight2', shape=(1,))
            # self.weight3 = self.params.get('weight3', shape=(1,))

            # depthwise separable convolution
            self._dws = HybridSequential()
            self._dws.add(Conv2D(channels, 3, strides=(1, 1), padding=(1, 1),
                                 groups=channels))
            self._dws.add(BatchNorm(momentum=momentum, epsilon=epsilon))
            self._dws.add(Activation('relu'))
            self._dws.add(Conv2D(channels, 1, strides=(1, 1), padding=(0, 0),
                                 groups=1))
            self._dws.add(BatchNorm(momentum=momentum, epsilon=epsilon))
            self._dws.add(Activation('relu'))

    def hybrid_forward(self, F, x1, x2, x3, weight1=None, weight2=None, weight3=None, epsilon=0.0001):
        '''
        In paper, weight* is ensured by applying a RELU activation Function - 이거 학습 안됨.
        '''
        # w1 = F.relu(weight1)
        # w2 = F.relu(weight2)
        # w3 = F.relu(weight3)
        #
        # total = w1 + w2 + w3 + epsilon
        # w1 = F.broadcast_div(w1, total)
        # w2 = F.broadcast_div(w2, total)
        # w3 = F.broadcast_div(w3, total)
        #
        # first = F.broadcast_mul(w1, x1)
        # second = F.broadcast_mul(w2, x2)
        # third = F.broadcast_mul(w3, x3)
        #
        first = x1
        second = x2
        third = x3
        x = first + second + third
        x = self._dws(x)

        return x


class Bifpn(HybridBlock):

    def __init__(self, layers=None, channels=None, version=None):
        super(Bifpn, self).__init__()

        self._layers = layers
        self._effbase = get_efficientbase(version=version)  # Efficient Net

        with self.name_scope():

            self._p7lateral = lateral_conv(channels, 1, 1, 0)
            self._p6lateral = lateral_conv(channels, 1, 1, 0)
            self._p5lateral = lateral_conv(channels, 1, 1, 0)
            self._p4lateral = lateral_conv(channels, 1, 1, 0)
            self._p3lateral = lateral_conv(channels, 1, 1, 0)

            self._p7 = HybridSequential()
            self._p6 = HybridSequential()
            self._p5 = HybridSequential()
            self._p4 = HybridSequential()
            self._p3 = HybridSequential()
            self._downsample = HybridSequential()

            for _ in range(layers):
                self._p7.add(HybridSequential())
                self._p6.add(HybridSequential())
                self._p5.add(HybridSequential())
                self._p4.add(HybridSequential())
                self._p3.add(HybridSequential())
                self._downsample.add(HybridSequential())

            for i in range(layers):
                self._p7[i].add(intermediateFF(channels))
                self._p6[i].add(intermediateFF(channels))
                self._p6[i].add(OutputFF(channels))
                self._p5[i].add(intermediateFF(channels))
                self._p5[i].add(OutputFF(channels))
                self._p4[i].add(intermediateFF(channels))
                self._p4[i].add(OutputFF(channels))
                self._p3[i].add(intermediateFF(channels))

                # down -> top
                self._downsample[i].add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
                self._downsample[i].add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
                self._downsample[i].add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
                self._downsample[i].add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    def hybrid_forward(self, F, x):

        # backbone - EfficientNet
        p3, p4, p5, p6, p7 = self._effbase(x)

        p3 = self._p3lateral(p3)
        p4 = self._p4lateral(p4)
        p5 = self._p5lateral(p5)
        p6 = self._p6lateral(p6)
        p7 = self._p7lateral(p7)

        for i in range(self._layers):
            # upsample
            p7_upsample = F.UpSampling(p7, scale=2, sample_type='nearest')
            p6_im = self._p6[i][0](p6, p7_upsample)

            p6_im_upsample = F.UpSampling(p6_im, scale=2, sample_type='nearest')
            p5_im = self._p5[i][0](p5, p6_im_upsample)

            p5_im_upsample = F.UpSampling(p5_im, scale=2, sample_type='nearest')
            p4_im = self._p4[i][0](p4, p5_im_upsample)

            p4_im_upsample = F.UpSampling(p4_im, scale=2, sample_type='nearest')
            p3 = self._p3[i][0](p3, p4_im_upsample)

            # downsample
            p3_downsample = self._downsample[i][0](p3)
            p4 = self._p4[i][1](p4, p4_im, p3_downsample)

            p4_downsample = self._downsample[i][1](p4)
            p5 = self._p5[i][1](p5, p5_im, p4_downsample)

            p5_downsample = self._downsample[i][2](p5)
            p6 = self._p6[i][1](p6, p6_im, p5_downsample)

            p6_downsample = self._downsample[i][3](p6)
            p7 = self._p7[i][0](p7, p6_downsample)

        return p3, p4, p5, p6, p7


# Constructor
def get_bifpn(version, ctx=mx.cpu(), dummy=False):
    """
    https://arxiv.org/pdf/1905.11946.pdf : EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    https://arxiv.org/pdf/1911.09070.pdf : EfficientDet: Scalable and Efficient Object Detection
    Parameters
    ----------
    version : int
        Numbers of layers. Options are 0, 1, 2, 3, 4, 5, 6, 7
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """

    # Box / Class Layer - 논문에 나온대로
    layers = [2, 3, 4, 5, 6, 7, 8, 8]
    channels = [64, 88, 112, 160, 224, 288, 384, 384]
    net = Bifpn(layers[version], channels[version], version)
    if not dummy:
        logging.info(f"EfficientBase_{version} weight init 완료")
        logging.info(f"BIFPN_{version} weight init 완료")
    net.initialize(ctx=ctx)
    return net


if __name__ == "__main__":

    ''' 입력 사이즈 
    D0 -> 512 x 512
    D1 -> 640 x 640
    D2 -> 768 x 768
    ------------------- 여기까지가 실시간으로 쓸만 할듯 -------------------
    D3 -> 896 x 896
    D4 -> 1024 x 1024
    D5 -> 1280 x 1280
    D6 -> 1408 x 1408
    D7 -> 1536 x 1536
    '''
    input_size = (512, 512)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    net = get_bifpn(0, ctx=mx.cpu())
    net.hybridize(active=True, static_alloc=True, static_shape=True)
    output = net(mx.nd.random_uniform(low=0, high=1, shape=(1, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
    print(f"< input size(height, width) : {input_size} >")
    print(f"< output number : {len(output)} >")
    for i, out in enumerate(output):
        print(f"({i + 1}) feature shape : {out.shape}")
    '''
    < input size(height, width) : (512, 512) >
    < output number : 5 >
    (1) feature shape : (1, 64, 64, 64)
    (2) feature shape : (1, 64, 32, 32)
    (3) feature shape : (1, 64, 16, 16)
    (4) feature shape : (1, 64, 8, 8)
    (5) feature shape : (1, 64, 4, 4)
    '''
