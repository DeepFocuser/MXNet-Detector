import logging
import os

import mxnet as mx
from mxnet.gluon import contrib
from mxnet.gluon.nn import HybridBlock, HybridSequential, Conv2D, Conv2DTranspose, BatchNorm, Activation

from core.model.backbone.ResNetV2 import get_resnet

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


class UpConvResNet(HybridBlock):

    def __init__(self, base=18,
                 deconv_channels=(256, 128, 64),
                 deconv_kernels=(4, 4, 4),
                 pretrained=True,
                 root=os.path.join(os.getcwd(), 'models'),
                 use_dcnv2=False,
                 ctx=mx.cpu()):

        mxnet_version = float(mx.__version__[0:3])
        if mxnet_version < 1.6:
            logging.error("please upgrade mxnet version above 1.6.x")
            raise EnvironmentError

        super(UpConvResNet, self).__init__()
        self._use_dcnv2 = use_dcnv2
        self._resnet = get_resnet(base, pretrained=pretrained, root=root, ctx=ctx)
        self._upconv = HybridSequential('')
        with self._upconv.name_scope():
            for channel, kernel in zip(deconv_channels, deconv_kernels):
                kernel, padding, output_padding = self._get_conv_argument(kernel)
                if self._use_dcnv2:
                    '''
                    in paper, we first change the channels of the three upsampling layers to
                    256, 128, 64, respectively, to save computation, we then add one 3 x 3 deformable convolutional layer
                    before each up-convolution layer with channel 256, 128, 64 
                    '''
                    assert hasattr(contrib.cnn, 'ModulatedDeformableConvolution'), \
                        "No ModulatedDeformableConvolution found in mxnet, consider upgrade to mxnet 1.6.0..."
                    self._upconv.add(contrib.cnn.ModulatedDeformableConvolution(channels=channel,
                                                                                kernel_size=3,
                                                                                strides=1,
                                                                                padding=1,
                                                                                use_bias=False,
                                                                                num_deformable_group=1))
                else:
                    self._upconv.add(Conv2D(channels=channel,
                                            kernel_size=3,
                                            strides=1,
                                            padding=1, use_bias=False))
                self._upconv.add(BatchNorm(momentum=0.9))
                self._upconv.add(Activation('relu'))
                self._upconv.add(Conv2DTranspose(channels=channel,
                                                 kernel_size=kernel,
                                                 strides=2,
                                                 padding=padding,
                                                 output_padding=output_padding,
                                                 use_bias=False,
                                                 weight_initializer=mx.init.Bilinear()))
                self._upconv.add(BatchNorm(momentum=0.9))
                self._upconv.add(Activation('relu'))

        self._upconv.initialize(ctx=ctx)
        logging.info(f"{self.__class__.__name__} weight init 완료")

    def _get_conv_argument(self, kernel):

        """Get the upconv configs using presets"""
        if kernel == 4:
            padding = 1
            output_padding = 0
        elif kernel == 3:
            padding = 1
            output_padding = 1
        elif kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError('Unsupported deconvolution kernel: {}'.format(kernel))
        return kernel, padding, output_padding

    def hybrid_forward(self, F, x):
        x = self._resnet(x)
        x = self._upconv(x)
        return x


def get_upconv_resnet(base=18, pretrained=False, root=os.path.join(os.getcwd(), 'models'), use_dcnv2=False,
                      ctx=mx.cpu()):
    net = UpConvResNet(base=base,
                       deconv_channels=(256, 128, 64),
                       deconv_kernels=(4, 4, 4),
                       pretrained=pretrained,
                       root=root,
                       use_dcnv2=use_dcnv2,
                       ctx=ctx)
    return net


if __name__ == "__main__":
    input_size = (960, 1280)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    net = get_upconv_resnet(base=50, pretrained=False, ctx=mx.cpu(), root=os.path.join(root, 'modelparam'),
                            use_dcnv2=False)
    net.hybridize(active=True, static_alloc=True, static_shape=True)
    output = net(mx.nd.random_uniform(low=0, high=1, shape=(1, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
    print(f"< input size(height, width) : {input_size} >")
    print(f"< output shape : {output.shape} >")
    '''
    < input size(height, width) : (512, 512) >
    < output shape : (1, 64, 128, 128) >
    '''
