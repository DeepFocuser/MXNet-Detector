# gluoncv에 있는 코드 참고

"""Encoder functions.
Encoders are used during training, which assign training targets.
"""
import mxnet as mx
from mxnet.gluon import Block


class BBoxCornerToCenter(Block):
    def __init__(self, axis=-1):
        super(BBoxCornerToCenter, self).__init__()
        self._axis = axis

    def forward(self, x):
        F = mx.nd
        xmin, ymin, xmax, ymax = F.split(x, axis=self._axis, num_outputs=4)
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + width / 2
        y_center = ymin + height / 2
        return x_center, y_center, width, height


class BoxEncoder(Block):
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(BoxEncoder, self).__init__()
        self._stds = stds
        self._means = means
        self._corner_to_center = BBoxCornerToCenter(axis=-1)

    def forward(self, matches, samples, anchors, gt_boxes):
        F = mx.nd
        gt_boxes = F.repeat(gt_boxes.reshape((0, 1, -1, 4)), repeats=matches.shape[1],
                            axis=1)  # (B, ALL Feature, Box number, 4)
        gt_boxes = F.split(gt_boxes, axis=-1, num_outputs=4,
                           squeeze_axis=True)  # 4 * (B, ALL Feature, Box number, 1) -> 4 * (B, ALL Feature, Box number)
        gt_boxed_element = [F.pick(gt_boxes[i], matches, axis=-1) for i in range(4)]  # 4*(B, ALL Feature)
        gt_boxes = F.stack(*gt_boxed_element, axis=-1)  # (B, ALL Feature, 4)
        gt_box_x, gt_box_y, gt_box_w, gt_box_h = self._corner_to_center(gt_boxes)

        anchor_x, anchor_y, anchor_w, anchor_h = self._corner_to_center(anchors)  #
        norm_x = (F.divide(F.subtract(gt_box_x, anchor_x), anchor_w) - self._means[0]) / self._stds[0]
        norm_y = (F.divide(F.subtract(gt_box_y, anchor_y), anchor_h) - self._means[1]) / self._stds[1]
        norm_w = (F.log(F.divide(gt_box_w, anchor_w)) - self._means[2]) / self._stds[2]
        norm_h = (F.log(F.divide(gt_box_h, anchor_h)) - self._means[3]) / self._stds[3]
        box_ids = F.concat(norm_x, norm_y, norm_w, norm_h, dim=-1)

        samples_repeat = F.repeat(samples.reshape((0, -1, 1)), repeats=4, axis=-1) > 0
        targets = F.where(samples_repeat, box_ids, F.zeros_like(box_ids))
        return targets


class ClassEncoder(Block):
    def __init__(self):
        super(ClassEncoder, self).__init__()

    def forward(self, matches, samples, gt_ids):
        F = mx.nd
        gt_ids_repeat = F.repeat(gt_ids.reshape((0, 1, -1)), axis=1, repeats=matches.shape[1])
        target_ids = F.pick(gt_ids_repeat, matches, axis=-1) + 1  # background 고려
        targets = F.where(samples > 0, target_ids, F.ones_like(target_ids) * -1)
        targets = F.where(samples < 0, F.zeros_like(targets), targets)
        return targets  # foreground class + background class 가 되어서 출력


# test
if __name__ == "__main__":
    from core import RetinaNet, DetectionDataset
    from core import MatchSampler
    import os

    input_size = (512, 512)
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), input_size=input_size,
                               image_normalization=True,
                               box_normalization=False)

    num_classes = dataset.num_class
    image, label, _ = dataset[0]

    net = RetinaNet(version=18,
                    input_size=input_size,
                    anchor_sizes=[32, 64, 128, 256, 512],
                    anchor_size_ratios=[1, pow(2, 1 / 3), pow(2, 2 / 3)],
                    anchor_aspect_ratios=[0.5, 1, 2],
                    num_classes=num_classes,  # foreground만
                    pretrained=False,
                    pretrained_path=os.path.join(root, "modelparam"),
                    anchor_box_offset=(0.5, 0.5),
                    anchor_box_clip=True,
                    ctx=mx.cpu())

    net.hybridize(active=True, static_alloc=True, static_shape=True)

    matchsampler = MatchSampler(foreground_iou_thresh=0.5, background_iou_thresh=0.4)
    classEncoder = ClassEncoder()
    boxEncoder = BoxEncoder(stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.))

    # batch 형태로 만들기
    image = image.expand_dims(axis=0)
    label = label.expand_dims(axis=0)

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    _, _, anchors = net(image)

    anchors, matches, samples = matchsampler(anchors, gt_boxes)
    cls_targets = classEncoder(matches, samples, gt_ids)
    box_targets = boxEncoder(matches, samples, anchors, gt_boxes)

    print(f"cls_targets shape : {cls_targets.shape}")
    print(f"box_targets shape : {box_targets.shape}")
    '''
    cls_targets shape : (1, 49104)
    box_targets shape : (1, 49104, 4)
    '''
