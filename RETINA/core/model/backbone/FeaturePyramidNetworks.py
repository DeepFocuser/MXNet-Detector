import logging
import os

import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential, Conv2D, BatchNorm, Activation, MaxPool2D

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def _conv3x3(channels, stride, in_channels):
    return Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                  use_bias=False, in_channels=in_channels)


class BasicBlockV2(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        if downsample:
            self.downsample = Conv2D(channels, 1, stride, use_bias=False,
                                     in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):

        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        return x + residual


class BottleneckV2(HybridBlock):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """

    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = BatchNorm()
        self.conv1 = Conv2D(channels // 4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = BatchNorm()
        self.conv2 = _conv3x3(channels // 4, stride, channels // 4)
        self.bn3 = BatchNorm()
        self.conv3 = Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        if downsample:
            self.downsample = Conv2D(channels, 1, stride, use_bias=False,
                                     in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual


class ResNetV2(HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    """

    def __init__(self, block, layers, channels, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            # pretrained weight 사용하려면, self.features = nn.HybridSequential(prefix='상관없음')을 사용하는 수밖에 없다.
            self.features = HybridSequential(prefix='')
            self.features.add(BatchNorm(scale=False, center=False))  # 의문점 하나 : 맨 앞에 왜 batch norm을???
            self.features.add(Conv2D(channels[0], 7, 2, 3, use_bias=False))
            self.features.add(BatchNorm())
            self.features.add(Activation('relu'))
            self.features.add(MaxPool2D(3, 2, 1))  # 4번째

            in_channels = channels[0]
            # 5(c2),6(c3),7(c4),8
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i + 1],
                                                   stride, i + 1, in_channels=in_channels))
                in_channels = channels[i + 1]
            self.features.add(BatchNorm())
            self.features.add(Activation('relu'))  # 10(c5)

            # FPN 논문에서, Output by each stage's last residual block
            # Top - Down
            self.lateral3 = Conv2D(channels=256, kernel_size=1, strides=1, padding=0, use_bias=False,
                                   prefix='lateral3')
            self.lateral4 = Conv2D(channels=256, kernel_size=1, strides=1, padding=0, use_bias=False,
                                   prefix='lateral4')
            self.lateral5 = Conv2D(channels=256, kernel_size=1, strides=1, padding=0, use_bias=False,
                                   prefix='lateral5')
            # extra convolution
            self.conv6 = Conv2D(channels=256, kernel_size=3, strides=2, padding=1, use_bias=False,
                                activation = "relu", prefix='extraconv1')
            self.conv7 = Conv2D(channels=256, kernel_size=3, strides=2, padding=1, activation="relu",
                                use_bias=False, prefix='extraconv2')

            #
            # self.upsampleconv_p5 = Conv2DTranspose(256, kernel_size=3, strides=2, padding=1,
            #                                                   output_padding=1,
            #                                                   activation='relu', use_bias=True, in_channels=0,
            #                                                   prefix='upsampleconv5')
            # self.upsampleconv_p4 = Conv2DTranspose(256, kernel_size=3, strides=2, padding=1,
            #                                                   output_padding=1,
            #                                                   activation='relu', use_bias=True, in_channels=0,
            #                                                   prefix='upsampleconv4')

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers - 1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):

        x = self.features[0:5](x)

        c2 = self.features[5](x)
        c3 = self.features[6](c2)
        c4 = self.features[7](c3)
        c5 = self.features[8:11](c4)
        p5 = self.lateral5(c5)

        # onnx를 위해 UpSampling 지양하기 - transposed convolution 으로 바꾸기
        p4 = F.UpSampling(p5, scale=2, sample_type='nearest') + self.lateral4(c4)  # 1/16
        p3 = F.UpSampling(p4, scale=2, sample_type='nearest') + self.lateral3(c3)  # 1/8

        # p4 = self.upsampleconv_p5(p5) + self.lateral4(c4)  # 1/16
        # p3 = self.upsampleconv_p4(p4) + self.lateral3(c3)  # 1/8

        p6 = self.conv6(c5)
        p7 = self.conv7(p6)

        return p3, p4, p5, p6, p7


def lateral_init(net, ctx):
    for param in net.collect_params().values():
        if param.name.find('lateral') != -1:
            param.initialize(init=mx.init.Normal(sigma=0.01), ctx=ctx)


def extraconv_init(net, ctx):
    for param in net.collect_params().values():
        if param.name.find('extraconv') != -1:
            param.initialize(init=mx.init.Normal(sigma=0.01), ctx=ctx)


def upsampleconv_init(net, ctx):
    for param in net.collect_params().values():
        if param.name.find('upsampleconv') != -1:
            param.initialize(init=mx.init.Normal(sigma=0.01), ctx=ctx)


######################################################################################################################
# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}
resnet_block_versions = {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}


# Constructor
def get_fpn_resnet(num_layers, pretrained=False, ctx=mx.cpu(),
                   root=os.path.join(os.getcwd(), 'models'), dummy=False):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnet_spec.keys()))

    block_type, layers, channels = resnet_spec[num_layers]
    block_class = resnet_block_versions[block_type]
    net = ResNetV2(block_class, layers, channels)
    if pretrained:
        from mxnet.gluon.model_zoo.model_store import get_model_file
        net.load_parameters(get_model_file(f'resnet{num_layers}_v2', root=root), ctx=ctx,
                            allow_missing=True,  # 무조건 True
                            ignore_extra=True)
        if not dummy:
            logging.info(f"resnet{num_layers} pretrained weight load 완료")
        lateral_init(net, ctx)
        extraconv_init(net, ctx)
        upsampleconv_init(net, ctx)
    else:
        if not dummy:
            logging.info(f"resnet{num_layers} weight init 완료")
        net.initialize(ctx=ctx)
    return net


if __name__ == "__main__":

    input_size = (2048, 2048)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    net = get_fpn_resnet(18, pretrained=False, ctx=mx.cpu(), root=os.path.join(root, 'modelparam'))
    net.hybridize(active=True, static_alloc=True, static_shape=True)
    output = net(mx.nd.random_uniform(low=0, high=1, shape=(2, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
    print(f"< input size(height, width) : {input_size} >")
    print(f"< output number : {len(output)} >")
    for i, out in enumerate(output):
        print(f"({i + 1}) feature shape : {out.shape}")
    '''
    < input size(height, width) : (512, 512) >
    < output number : 5 >
    (1) feature shape : (2, 256, 64, 64)
    (2) feature shape : (2, 256, 32, 32)
    (3) feature shape : (2, 256, 16, 16)
    (4) feature shape : (2, 256, 8, 8)
    (5) feature shape : (2, 256, 4, 4)
    '''
