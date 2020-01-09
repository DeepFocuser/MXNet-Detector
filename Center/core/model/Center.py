import logging
import os
from collections import OrderedDict

import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential, Conv2D, Activation

from core.model.backbone.UpConvResNetV2 import get_upconv_resnet

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


class CenterNet(HybridBlock):

    def __init__(self, base=18, heads=OrderedDict(), head_conv_channel=64, pretrained=True,
                 root=os.path.join(os.getcwd(), 'models'), use_dcnv2=False, ctx=mx.cpu()):
        super(CenterNet, self).__init__()

        with self.name_scope():
            self._base_network = get_upconv_resnet(base=base, pretrained=pretrained, root=root, use_dcnv2=use_dcnv2,
                                                   ctx=ctx)
            self._heads = HybridSequential('heads')
            for name, values in heads.items():
                head = HybridSequential(name)
                num_output = values['num_output']
                bias = values.get('bias', 0.0)
                head.add(Conv2D(head_conv_channel, kernel_size=(3, 3), padding=(1, 1), use_bias=True))
                head.add(Activation('relu'))
                head.add(Conv2D(num_output, kernel_size=(1, 1),
                                use_bias=True,
                                bias_initializer=mx.init.Constant(bias)))
                self._heads.add(head)
        self._heads.initialize(ctx=ctx)

    def hybrid_forward(self, F, x):
        feature = self._base_network(x)
        heatmap, offset, wh = [head(feature) for head in self._heads]
        heatmap = F.sigmoid(heatmap)
        return heatmap, offset, wh


if __name__ == "__main__":
    input_size = (768, 1280)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    '''
    heatmap의 bias가 -2.19 인 이유는??? retinanet의 식과 같은데... 흠..
    For the final conv layer of the classification subnet, we set the bias initialization to b = − log((1 − π)/π),
    where π specifies that at the start of training every anchor should be labeled as foreground with confidence of ∼π.
    We use π = .01 in all experiments, although results are robust to the exact value. As explained in §3.3, 
    this initialization prevents the large number of background anchors from generating a large, 
    destabilizing loss value in the first iteration of training
    '''
    net = CenterNet(base=18,
                    heads=OrderedDict([
                        ('heatmap', {'num_output': 5, 'bias': -2.19}),
                        ('offset', {'num_output': 2}),
                        ('wh', {'num_output': 2})
                    ]),
                    head_conv_channel=64,
                    pretrained=False,
                    root=os.path.join(root, 'models'),
                    use_dcnv2=False, ctx=mx.cpu())

    net.hybridize(active=True, static_alloc=True, static_shape=True)
    heatmap, offset, wh = net(
        mx.nd.random_uniform(low=0, high=1, shape=(1, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
    print(f"< input size(height, width) : {input_size} >")
    print(f"heatmap prediction shape : {heatmap.shape}")
    print(f"offset prediction shape : {offset.shape}")
    print(f"width height prediction shape : {wh.shape}")
    '''
    heatmap prediction shape : (1, 3, 128, 128)
    offset prediction shape : (1, 2, 128, 128)
    width height prediction shape : (1, 2, 128, 128)
    '''
