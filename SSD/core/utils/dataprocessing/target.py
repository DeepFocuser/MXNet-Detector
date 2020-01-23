import mxnet as mx
from mxnet.gluon import Block

from core.utils.dataprocessing.targetFunction.encoder import ClassEncoder, BoxEncoder
from core.utils.dataprocessing.targetFunction.matching import MatchSampler


class TargetGenerator(Block):

    def __init__(self, foreground_iou_thresh=0.5, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(TargetGenerator, self).__init__()
        self._matchsampler = MatchSampler(foreground_iou_thresh=foreground_iou_thresh)
        self._cls_encoder = ClassEncoder()
        self._box_encoder = BoxEncoder(stds=stds, means=means)

    def forward(self, anchors, gt_boxes, gt_ids):
        """Generate training targets."""
        anchors_corner, matches, samples = self._matchsampler(anchors, gt_boxes)
        cls_targets = self._cls_encoder(matches, samples, gt_ids)
        box_targets = self._box_encoder(matches, samples, anchors_corner, gt_boxes)
        return cls_targets, box_targets


# test
if __name__ == "__main__":
    from core import SSD_VGG16, SSDTrainTransform, DetectionDataset
    import os

    input_size = (512, 512)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = SSDTrainTransform(input_size[0], input_size[1], make_target=False)
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), transform=transform)
    num_classes = dataset.num_class
    image, label, _, _, _ = dataset[0]
    label = mx.nd.array(label)

    net = SSD_VGG16(version=512, input_size=input_size,
                    # box_sizes=[21, 45, 101.25, 157.5, 213.75, 270, 326.25],
                    box_sizes=[21, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
                    # box_ratios=[[1, 2, 0.5]] +  # conv4_3
                    #           [[1, 2, 0.5, 3, 1.0 / 3]] * 3 +  # conv7, conv8_2, conv9_2
                    #           [[1, 2, 0.5]] * 2,  # conv10_2, conv11_2
                    box_ratios=[[1, 2, 0.5]] +  # conv4_3
                               [[1, 2, 0.5, 3, 1.0 / 3]] * 4 +  # conv7, conv8_2, conv9_2, conv10_2
                               [[1, 2, 0.5]] * 2,  # conv11_2, conv12_2
                    num_classes=num_classes,
                    pretrained=False,
                    pretrained_path=os.path.join(root, "modelparam"),
                    anchor_box_offset=(0.5, 0.5),
                    anchor_box_clip=True,
                    ctx=mx.cpu())

    net.hybridize(active=True, static_alloc=True, static_shape=True)

    targetgenerator = TargetGenerator(foreground_iou_thresh=0.5, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.))

    # batch 형태로 만들기
    image = image.expand_dims(axis=0)
    label = label.expand_dims(axis=0)

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    _, _, anchors = net(image)
    cls_targets, box_targets = targetgenerator(anchors, gt_boxes, gt_ids)
    print(f"cls_targets shape : {cls_targets.shape}")
    print(f"box_targets shape : {box_targets.shape}")
    '''
    cls_targets shape : (1, 24564)
    box_targets shape : (1, 24564, 4)
    '''
