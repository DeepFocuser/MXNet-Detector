import mxnet as mx
from mxnet.gluon import Block

from core.utils.dataprocessing.targetFunction.encodedynamic import Encoderdynamic
from core.utils.dataprocessing.targetFunction.encoderfix import Encoderfix
from core.utils.dataprocessing.targetFunction.matching import Matcher


class TargetGenerator(Block):

    def __init__(self, ignore_threshold=0.5, dynamic=False, from_sigmoid=False):
        super(TargetGenerator, self).__init__()
        self._matcher = Matcher()
        self._from_sigmoid = from_sigmoid

        '''
        https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/61 : 이슈
        ignore label을 만드는 방식이다.
        dynamic 방식은 요즘? Darknet에 구현되어 있다고 한다. 
        yolov3에 있는 방식은 target과 anchor을 이용한 방식,
        dynamic방식은 네트워크의 pred와 targer을 이용한 방식이다.
        논문에는 target과 anchor을 이용한 방식으로 되어있는데...
        정작 논문 저자의 Darknet코드에는 네트워크의 pred와 targer을 이용한 방식이 구현되어있다는...
        '''
        self._dynamic = dynamic
        if dynamic:
            self._encoder = Encoderdynamic(ignore_threshold=ignore_threshold, from_sigmoid=from_sigmoid)
        else:
            self._encoder = Encoderfix(ignore_threshold=ignore_threshold)

    def forward(self, outputs, anchors, gt_boxes, gt_ids, input_size):
        matches, ious = self._matcher(anchors, gt_boxes)
        if self._dynamic:
            return self._encoder(matches, ious, outputs, anchors, gt_boxes, gt_ids, input_size)
        else:
            return self._encoder(matches, ious, outputs, anchors, gt_boxes, gt_ids, input_size)


# test
if __name__ == "__main__":
    from core import Yolov3, YoloTrainTransform, DetectionDataset
    import os

    input_size = (416, 416)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = YoloTrainTransform(input_size[0], input_size[1])
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), transform=transform)
    num_classes = dataset.num_class

    image, label, _ = dataset[0]

    net = Yolov3(Darknetlayer=53,
                 input_size=input_size,
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=5,  # foreground만
                 pretrained=False,
                 pretrained_path=os.path.join(root, "modelparam"),
                 ctx=mx.cpu())
    net.hybridize(active=True, static_alloc=True, static_shape=True)

    targetgenerator = TargetGenerator(ignore_threshold=0.5, dynamic=True, from_sigmoid=False)

    # batch 형태로 만들기
    image = image.expand_dims(axis=0)
    label = label.expand_dims(axis=0)

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    output1, output2, output3, anchor1, anchor2, anchor3, _, _, _, _, _, _ = net(image)
    xcyc_targets, wh_targets, objectness, class_targets, weights = targetgenerator([output1, output2, output3],
                                                                                   [anchor1, anchor2, anchor3],
                                                                                   gt_boxes,
                                                                                   gt_ids,
                                                                                   input_size)

    print(f"< input size(height, width) : {input_size} >")
    print(f"xcyc_targets shape : {xcyc_targets.shape}")
    print(f"wh_targets shape : {wh_targets.shape}")
    print(f"objectness shape : {objectness.shape}")
    print(f"class_targets shape : {class_targets.shape}")
    print(f"weights shape : {weights.shape}")
    '''
    < input size(height, width) : (416, 416) >
    xcyc_targets shape : (1, 10647, 2)
    wh_targets shape : (1, 10647, 2)
    objectness shape : (1, 10647, 1)
    class_targets shape : (1, 10647)
    weights shape : (1, 10647, 2)
    '''
