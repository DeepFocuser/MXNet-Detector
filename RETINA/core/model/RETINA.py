import numpy as np

from core.model.backbone.FeaturePyramidNetworks import *


class RetinaAnchorGenerator(HybridBlock):

    def __init__(self, index=0,
                 input_size=(512, 512),
                 feature_size=(64, 64),
                 anchor_size=32,
                 anchor_size_ratios=[1, pow(2, 1 / 3), pow(2, 2 / 3)],
                 anchor_aspect_ratios=[0.5, 1, 2],
                 box_offset=(0.5, 0.5), box_clip=False, alloc_size=[256, 256]):
        super(RetinaAnchorGenerator, self).__init__()
        self._height = input_size[0]
        self._width = input_size[1]
        self._box_clip = box_clip

        # 아래와 같이 등록해주면 hybrid_forward에 anchors로서 들어감
        with self.name_scope():
            self.anchors = self.params.get_constant(name='anchor_%d' % (index),
                                                    value=self._generate_anchors(feature_size=feature_size,
                                                                                 anchor_size=anchor_size,
                                                                                 anchor_size_ratios=anchor_size_ratios,
                                                                                 anchor_aspect_ratios=anchor_aspect_ratios,
                                                                                 box_offset=box_offset,
                                                                                 alloc_size=alloc_size))

    def _generate_anchors(self, feature_size=None, anchor_size=None, anchor_size_ratios=None, anchor_aspect_ratios=None,
                          box_offset=None, alloc_size=None):
        # cx, xy, w, h 순으로 붙임!!!

        y_range = max(feature_size[0], alloc_size[0])
        x_range = max(feature_size[1], alloc_size[1])
        step_y = self._height / feature_size[0]
        step_x = self._width / feature_size[1]
        anchor_size_ratios_width = np.multiply(anchor_size, np.array(anchor_size_ratios))
        anchor_size_ratios_height = np.multiply(anchor_size, np.array(anchor_size_ratios))

        anchors = []
        for y in range(y_range):
            for x in range(x_range):
                for nasr_w, nasr_h in zip(anchor_size_ratios_width, anchor_size_ratios_height):
                    cy = (y + box_offset[0]) * step_y
                    cx = (x + box_offset[1]) * step_x
                    for asr in anchor_aspect_ratios:
                        sr = np.sqrt(asr)
                        w = nasr_w * sr
                        h = nasr_h / sr
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
                 use_bias=False,
                 in_channels=0,
                 weight_initializer="zeros",
                 bias_initializer='zeros'):
        super(ConvPredictor, self).__init__()
        with self.name_scope():
            self.predictor = Conv2D(
                num_channel, kernel, strides=stride, padding=pad,
                activation=activation, use_bias=use_bias, in_channels=in_channels,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer)

    def hybrid_forward(self, F, x):
        return self.predictor(x)


class RetinaNet(HybridBlock):

    def __init__(self, version=18,
                 input_size=(512, 512),
                 anchor_sizes=[32, 64, 128, 256, 512],
                 anchor_size_ratios=[1, pow(2, 1 / 3), pow(2, 2 / 3)],
                 anchor_aspect_ratios=[0.5, 1, 2],
                 num_classes=1,  # foreground만
                 pretrained=False,
                 pretrained_path="modelparam",
                 anchor_box_offset=(0.5, 0.5),
                 anchor_box_clip=True,
                 alloc_size=[256, 256],
                 ctx=mx.cpu()):
        super(RetinaNet, self).__init__()

        if version not in [18, 34, 50, 101, 152]:
            raise ValueError

        feature_sizes = []
        fpn_output = get_fpn_resnet(version, ctx=mx.cpu(), dummy=True)(
            mx.nd.random_uniform(low=0, high=1, shape=(1, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
        for fpn in fpn_output:
            feature_sizes.append(fpn.shape[2:])  # h, w

        self._fpn_resnet = get_fpn_resnet(version, pretrained=pretrained, ctx=ctx, root=pretrained_path)
        self._num_classes = num_classes

        with self.name_scope():

            self._class_subnet = HybridSequential()
            self._box_subnet = HybridSequential()
            self._anchor_generators = HybridSequential()
            for _ in range(3):
                self._class_subnet.add(ConvPredictor(num_channel=256,
                                                     kernel=(3, 3),
                                                     pad=(1, 1),
                                                     stride=(1, 1),
                                                     activation='relu',
                                                     use_bias=True,
                                                     in_channels=0,
                                                     weight_initializer=mx.init.Normal(0.01),
                                                     bias_initializer='zeros'
                                                     ))
                self._box_subnet.add(ConvPredictor(num_channel=256,
                                                   kernel=(3, 3),
                                                   pad=(1, 1),
                                                   stride=(1, 1),
                                                   activation='relu',
                                                   use_bias=True,
                                                   in_channels=0,
                                                   weight_initializer=mx.init.Normal(0.01),
                                                   bias_initializer='zeros'
                                                   ))

            '''
            bias_initializer=mx.init.Constant(-np.log((1-0.01)/0.01)?
            논문에서,
             For the final convlayer of the classification subnet, we set the bias initialization to 
             b = − log((1 − π)/π), where π specifies that at that at the start of training every anchor should be 
             labeled as foreground with confidence of ∼π. We use π = .01 in all experiments, 
             although results are robust to the exact value. As explained in section 3.3, 
             this initialization prevents the large number of background anchors from generating a large, 
             destabilizing loss value in the first iteration of training.
             
             -> 초기에 class subnet 마지막 bias를 b = − log((1 − π)/π)로 설정함으로써, 
             모든  anchor를 0.01 값을 가지는 foreground로 만들어버리는 초기화 방법
             거의 대부분인 background anchor가 첫 번째 학습에서 불안정한 loss 값을 가지는 것을 방지해준다함.
            '''
            prior = 0.01
            self._class_subnet.add(
                ConvPredictor(num_channel=num_classes * len(anchor_size_ratios) * len(anchor_aspect_ratios),
                              kernel=(3, 3),
                              pad=(1, 1),
                              stride=(1, 1),
                              activation=None,
                              use_bias=True,
                              in_channels=0,
                              weight_initializer=mx.init.Normal(0.01),
                              bias_initializer=mx.init.Constant(-np.log((1 - prior) / prior))
                              ))

            self._box_subnet.add(ConvPredictor(num_channel=4 * len(anchor_size_ratios) * len(anchor_aspect_ratios),
                                               kernel=(3, 3),
                                               pad=(1, 1),
                                               stride=(1, 1),
                                               activation=None,
                                               use_bias=True,
                                               in_channels=0,
                                               weight_initializer=mx.init.Normal(0.01),
                                               bias_initializer='zeros'
                                               ))

            for index, feature_size, anchor_size in zip(range(len(feature_sizes)), feature_sizes, anchor_sizes):
                self._anchor_generators.add(RetinaAnchorGenerator(index=index,
                                                                  input_size=input_size,
                                                                  feature_size=feature_size,
                                                                  anchor_size=anchor_size,
                                                                  anchor_size_ratios=anchor_size_ratios,
                                                                  anchor_aspect_ratios=anchor_aspect_ratios,
                                                                  box_offset=anchor_box_offset,
                                                                  box_clip=anchor_box_clip,
                                                                  alloc_size=(alloc_size[0] // (2 ** index),
                                                                              alloc_size[1] // (2 ** index))))

        self._class_subnet.initialize(ctx=ctx)
        self._box_subnet.initialize(ctx=ctx)
        self._anchor_generators.initialize(ctx=ctx)
        logging.info(f"{self.__class__.__name__} Head weight init 완료")

    def hybrid_forward(self, F, x):

        # class, box prediction
        # self._fpn_resnet(x)  # p3, p4, p5, p6, p7
        # (batch, height, width, class) -> (batch, -1)
        cls_preds = [F.flatten(data=F.transpose(data=self._class_subnet(fpn_feature), axes=(0, 2, 3, 1)))
                     for fpn_feature in self._fpn_resnet(x)]
        # (batch, height, width, 4) -> (batch, -1)
        box_preds = [F.flatten(data=F.transpose(data=self._box_subnet(fpn_feature), axes=(0, 2, 3, 1)))
                     for fpn_feature in self._fpn_resnet(x)]
        anchors = [anchor_generator(fpn_feature) for fpn_feature, anchor_generator in
                   zip(self._fpn_resnet(x), self._anchor_generators)]

        cls_preds = F.reshape(data=F.concat(*cls_preds, dim=-1), shape=(0, -1, self._num_classes))
        box_preds = F.reshape(data=F.concat(*box_preds, dim=-1), shape=(0, -1, 4))
        anchors = F.reshape(F.concat(*anchors, dim=0), shape=(1, -1, 4))

        return cls_preds, box_preds, anchors


if __name__ == "__main__":
    input_size = (512, 512)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    net = RetinaNet(version=18,
                    input_size=input_size,
                    anchor_sizes=[32, 64, 128, 256, 512],
                    anchor_size_ratios=[1, pow(2, 1 / 3), pow(2, 2 / 3)],
                    anchor_aspect_ratios=[0.5, 1, 2],
                    num_classes=7,  # foreground만
                    pretrained=False,
                    pretrained_path=os.path.join(root, "modelparam"),
                    anchor_box_offset=(0.5, 0.5),
                    anchor_box_clip=True,
                    alloc_size=[256, 256],
                    ctx=mx.cpu())
    net.hybridize(active=True, static_alloc=True, static_shape=True)
    cls_preds, box_preds, anchors = net(
        mx.nd.random_uniform(low=0, high=1, shape=(2, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
    print(f"< input size(height, width) : {input_size} >")
    print(f"class prediction shape : {cls_preds.shape}")
    print(f"box prediction shape : {box_preds.shape}")
    print(f"anchor shape : {anchors.shape}")
    '''
    < input size(height, width) : (512, 512) >
    class prediction shape : (2, 49104, 7)
    box prediction shape : (2, 49104, 4)
    anchor shape : (1, 785664, 4)
    '''
