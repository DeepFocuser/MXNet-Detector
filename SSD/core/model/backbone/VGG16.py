import logging
import os

import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential, Conv2D, Activation

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

# gluoncv 내부 코드를 참고함
# original caffe prototxt : https://gist.github.com/tanshen/20cb9c6654f77b5e51f00ba0e08b1da9 참고

'''
fc6, fc7 -> convolution layer로 대체
pool5(2x2, stride 2) -> pool5(3x3. stride 1)
atrous algorithm사용 
dropout, fc8 layer 제거

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
'''

# vgg16's layer list, filter list
spec = ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512])

extra_spec = {
    300: [((256, 1, 1, 0), (512, 3, 2, 1)),  # channels, kernel, strides, padding / channels, kernel, strides, padding
          ((128, 1, 1, 0), (256, 3, 2, 1)),
          ((128, 1, 1, 0), (256, 3, 1, 0)),
          ((128, 1, 1, 0), (256, 3, 1, 0))],

    512: [((256, 1, 1, 0), (512, 3, 2, 1)),
          ((128, 1, 1, 0), (256, 3, 2, 1)),
          ((128, 1, 1, 0), (256, 3, 2, 1)),
          ((128, 1, 1, 0), (256, 3, 2, 1)),
          ((128, 1, 1, 0), (256, 4, 1, 1))],
}

'''
이것을  하는 이유? 
https://arxiv.org/pdf/1506.04579.pdf (3.3 L2 NORMALIZATION LAYER)
위 논문에서...
In our work, we apply L2-norm and learn the scale parameter for each channel before using the
feature for classification, which leads to more stable training.

SSD 논문에서...
conv4_3 has a different feature scale compared to the other layer, we use the L2 normalization technique
to scale th feature norm at each location in the feature map to 20 and learn the scale during back propagation
'''


# tensorRT에서 지원을 안함...
class Normalize(HybridBlock):

    def __init__(self, n_channel, initial=20, eps=1e-5):
        super(Normalize, self).__init__()
        self.eps = eps

        # __setattr__(self, name, value)
        # self._reg_params[name] = value # name=scale , value= self.params.get(...) 이 된다.
        with self.name_scope():
            self.scale = self.params.get('normalize_scale', shape=(1, n_channel, 1, 1),
                                         init=mx.init.Constant(initial))

    # self.scale이므로 반드시 아래에 있는 인자 이름도 scale이어야 한다.(내부코드를 보니 __setattr__를 사용해서 name 얻고, 
    # params = {i: j.data(ctx) for i, j in self._reg_params.items()} 이렇게 구한다음  **param으로 hybrid_forward에 넘겨버린다.
    '''
    이런식으로 동작함 - 함수 f의 scale의 이름은 반드시 scale 이어야함 
    def f(temp,scale):
        print(scale)
    f(3, scale=4) # scale로 보냈음.
    '''

    def hybrid_forward(self, F, x, scale):
        # onnx mode='channel' 만 지원 - tensorRT에서 지원을 안함.
        x = F.L2Normalization(x, mode='channel', eps=self.eps)
        # onnx 지원
        return F.broadcast_mul(x, scale)


class VGGAtrousBase(HybridBlock):

    def __init__(self, layers, filters):
        super(VGGAtrousBase, self).__init__()
        with self.name_scope():

            '''
            # caffe에서 가져온 pre-trained weights를 사용하기 때문에, 아래와 같은 init_scale가 필요하다고 함
            -> caffe의 pre-trained model은 입력 scale이 0 ~ 255임 
            '''
            init_scale = mx.nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) * 255
            self.init_scale = self.params.get_constant('init_scale', init_scale)

            # layers : [2, 2, 3, 3, 3], filters [64, 128, 256, 512, 512])
            self.stages = HybridSequential()
            for layer, filter in zip(layers, filters):
                stage = HybridSequential(prefix='')
                with stage.name_scope():
                    for _ in range(layer):
                        stage.add(Conv2D(filter,
                                         kernel_size=3,
                                         padding=1,
                                         weight_initializer=mx.init.Xavier(rnd_type='gaussian',
                                                                           factor_type='out', magnitude=3),
                                         bias_initializer='zeros'))
                        stage.add(Activation('relu'))
                self.stages.add(stage)

            # fc6, fc7 to dilated convolution layer - hybrid_forward에서 pooling 진행
            stage = HybridSequential(prefix='dilated_')
            with stage.name_scope():
                # conv6(fc6) - dilated
                stage.add(Conv2D(1024, kernel_size=3,
                                 padding=6,
                                 dilation=6,
                                 weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='out',
                                                                   magnitude=3),
                                 bias_initializer='zeros'))
                stage.add(Activation('relu'))

                # conv7(fc7)
                stage.add(Conv2D(1024, kernel_size=1,
                                 weight_initializer=mx.init.Xavier(rnd_type='gaussian', factor_type='out',
                                                                   magnitude=3),
                                 bias_initializer='zeros'))
                stage.add(Activation('relu'))

            self.stages.add(stage)
            self.norm4 = Normalize(n_channel=filters[3], initial=20, eps=1e-5)


class VGGAtrousExtractor(VGGAtrousBase):

    def __init__(self, layers, filters, extras):
        super(VGGAtrousExtractor, self).__init__(layers, filters)

        '''
        extra_spec = {
        300: [((256, 1, 1, 0), (512, 3, 2, 1)),
              ((128, 1, 1, 0), (256, 3, 2, 1)),
              ((128, 1, 1, 0), (256, 3, 1, 0)),
              ((128, 1, 1, 0), (256, 3, 1, 0))],
    
        512: [((256, 1, 1, 0), (512, 3, 2, 1)),
              ((128, 1, 1, 0), (256, 3, 2, 1)),
              ((128, 1, 1, 0), (256, 3, 2, 1)),
              ((128, 1, 1, 0), (256, 3, 2, 1)),
              ((128, 1, 1, 0), (256, 4, 1, 1))],
        '''
        # out_height = floor((height+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0])+1
        with self.name_scope():
            self.extras = HybridSequential()
            for i, config in enumerate(extras):
                extra = HybridSequential(prefix='extra%d_' % (i))
                with extra.name_scope():
                    for channels, kernel, strides, padding in config:
                        extra.add(Conv2D(channels=channels,
                                         kernel_size=kernel,
                                         strides=strides,
                                         padding=padding,
                                         weight_initializer=mx.init.Xavier(rnd_type='gaussian',
                                                                           factor_type='out', magnitude=3),
                                         bias_initializer='zeros'))
                        extra.add(Activation('relu'))
                self.extras.add(extra)

    '''
    ONNX 사용시 Pooling 주의 할 것. - https://github.com/onnx/onnx/issues/549
    ONNX currently doesn't support pooling_convention. This might lead to shape or accuracy issues.
    '''

    def hybrid_forward(self, F, x, init_scale):

        # onnx 지원함
        x = F.broadcast_mul(x, init_scale)

        outputs = []

        # conv4_3, conv7(fc7)
        for stage in self.stages[:3]:
            x = stage(x)
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                          pooling_convention='valid')
        x = self.stages[3](x)
        norm = self.norm4(x)
        outputs.append(norm)  # conv4_3
        x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                      pooling_convention='valid')

        x = self.stages[4](x)  # pool5 from 2 x 2 -s2 to 3 x 3 - s1
        # pooling_convention = full, which is compatible with Caffe:
        x = F.Pooling(x, pool_type='max', kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      pooling_convention='valid')

        # atrous convolution 적용
        x = self.stages[5](x)  # conv7(fc7)
        outputs.append(x)

        # inputsize 300 -> conv8_2, conv9_2, conv10_2, conv11_2
        # inputsize 512 -> conv8_2, conv9_2, conv10_2, conv11_2, conv12_2
        for extra in self.extras:
            x = extra(x)
            outputs.append(x)
        return outputs


def VGG16(version=512, pretrained=False, ctx=mx.cpu(), root="modelparam", dummy=False):
    if version not in [300, 512]:
        raise ValueError

    layers, filters = spec
    extras = extra_spec[version]
    net = VGGAtrousExtractor(layers, filters, extras)
    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        net.initialize(ctx=ctx)
        net.load_parameters(get_model_file('vgg16_atrous', tag=pretrained, root=root),
                            ctx=ctx,
                            ignore_extra=True,
                            allow_missing=True)
        '''
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        '''
        if not dummy:
            logging.info(f"{VGG16.__name__} pretrained weight load 완료")
        return net
    else:
        net.initialize(ctx=ctx)
        if not dummy:
            logging.info(f"{VGG16.__name__} weight init 완료")
        return net


if __name__ == "__main__":

    input_size = (2048, 2048)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    net = VGG16(version=512, pretrained=False, ctx=mx.cpu(), root=os.path.join(root, "modelparam"))
    net.hybridize(active=True, static_alloc=True, static_shape=True)
    output = net(mx.nd.random_uniform(low=0, high=1, shape=(8, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
    print(f"< input size : {input_size} >")
    print(f"< output number : {len(output)} >")
    for i, out in enumerate(output):
        print(f"({i + 1}) feature shape : {out.shape}")
    '''
    (1) feature shape : (8, 512, 256, 256)
    (2) feature shape : (8, 1024, 128, 128)
    (3) feature shape : (8, 512, 64, 64)
    (4) feature shape : (8, 256, 32, 32)
    (5) feature shape : (8, 256, 16, 16)
    (6) feature shape : (8, 256, 8, 8)
    (7) feature shape : (8, 256, 7, 7)
    '''
