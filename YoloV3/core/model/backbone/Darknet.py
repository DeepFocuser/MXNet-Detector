import logging
import os

import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential, Conv2D, BatchNorm, LeakyReLU

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def _conv2d(channel, kernel, padding, stride, norm_layer=BatchNorm):
    """A common conv-bn-leakyrelu cell"""
    cell = HybridSequential(prefix='')
    cell.add(Conv2D(channel, kernel_size=kernel,
                    strides=stride, padding=padding, use_bias=False))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9))
    cell.add(LeakyReLU(0.1))
    return cell


class DarknetBasicBlockV3(HybridBlock):
    """Darknet Basic Block. Which is a 1x1 reduce conv followed by 3x3 conv.

    Parameters
    ----------
    channel : int
        Convolution channels for 1x1 conv.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, channel, norm_layer=BatchNorm):
        super(DarknetBasicBlockV3, self).__init__()
        self.body = HybridSequential(prefix='')
        # 1x1 reduce
        self.body.add(_conv2d(channel, 1, 0, 1, norm_layer=norm_layer))
        # 3x3 conv expand
        self.body.add(_conv2d(channel * 2, 3, 1, 1, norm_layer=norm_layer))

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x, *args):
        residual = x
        x = self.body(x)
        return x + residual


# Darknet spec
specs = {53: ([1, 2, 8, 8, 4], [32, 64, 128, 256, 512, 1024])}


class DarknetV3(HybridBlock):
    """Darknet v3.
    Parameters
    ----------
    layers : iterable
        Description of parameter `layers`.
    channels : iterable
        Description of parameter `channels`.
    classes : int, default is 1000
        Number of classes, which determines the dense layer output channels.

    Attributes
    ----------
    features : mxnet.gluon.nn.HybridSequential
        Feature extraction layers.
    output : mxnet.gluon.nn.Dense
        A classes(1000)-way Fully-Connected Layer.
    """

    def __init__(self, layers, channels,
                 norm_layer=BatchNorm):
        super(DarknetV3, self).__init__()
        assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
                len(channels), len(layers)))

        with self.name_scope():
            self.features = HybridSequential()
            # first 3x3 conv /  channel, kernel, padding, stride, norm_layer
            self.features.add(_conv2d(channels[0], 3, 1, 1,
                                      norm_layer=norm_layer))

            # layer : [1, 2, 8, 8, 4] / channel : [32, 64, 128, 256, 512, 1024]
            for nlayer, channel in zip(layers, channels[1:]):
                assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
                # add downsample conv with stride=2
                self.features.add(_conv2d(channel, 3, 1, 2,
                                          norm_layer=norm_layer))
                # add nlayer basic blocks
                for _ in range(nlayer):
                    self.features.add(DarknetBasicBlockV3(channel // 2,
                                                          norm_layer=BatchNorm))

    def hybrid_forward(self, F, x):

        # darknet config 기준
        feature_36 = self.features[:15](x)
        feature_61 = self.features[15:24](feature_36)
        feature_74 = self.features[24:](feature_61)
        return feature_36, feature_61, feature_74


def get_darknet(num_layers, pretrained=False, ctx=mx.cpu(),
                root=os.path.join(os.getcwd(), 'models'), dummy=False, **kwargs):
    if num_layers not in [53]:
        raise ValueError

    layers, channels = specs[num_layers]
    net = DarknetV3(layers, channels, **kwargs)

    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        net.load_parameters(get_model_file(
            f"darknet{num_layers}", tag=pretrained, root=root), allow_missing=True, ignore_extra=True, ctx=ctx)
        if not dummy:
            logging.info(f"Darknet{num_layers} pretrained weight load 완료")
    else:
        net.initialize(ctx=ctx)
        if not dummy:
            logging.info(f"Darknet{num_layers} weight init 완료")
    return net


if __name__ == "__main__":

    input_size = (416, 416)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    net = get_darknet(53, pretrained=False, ctx=mx.cpu(), root=os.path.join(root, 'modelparam'))
    net.hybridize(active=True, static_alloc=True, static_shape=True)
    output = net(mx.nd.random_uniform(low=0, high=1, shape=(1, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
    print(f"< input size(height, width) : {input_size} >")
    print(f"< output number : {len(output)} >")
    for i, out in enumerate(output):
        print(f"({i + 1}) feature shape : {out.shape}")

    '''
    < input size(height, width) : (416, 416) >
    < output number : 3 >
    (1) feature shape : (1, 256, 52, 52)
    (2) feature shape : (1, 512, 26, 26)
    (3) feature shape : (1, 1024, 13, 13)
    '''
