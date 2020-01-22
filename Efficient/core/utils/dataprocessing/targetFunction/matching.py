# gluoncv에 있는 코드 참고

import mxnet as mx
from mxnet.gluon import Block


class BBoxCenterToCorner(Block):

    def __init__(self, axis=-1):
        super(BBoxCenterToCorner, self).__init__()
        self._axis = axis

    def forward(self, x):
        F = mx.nd
        x, y, width, height = F.split(x, axis=self._axis, num_outputs=4)
        half_w = width / 2
        half_h = height / 2
        xmin = x - half_w
        ymin = y - half_h
        xmax = x + half_w
        ymax = y + half_h
        return F.concat(xmin, ymin, xmax, ymax, dim=self._axis)


class MatchSampler(Block):

    def __init__(self, foreground_iou_thresh=0.5, background_iou_thresh=None):
        super(MatchSampler, self).__init__()
        self._centertocorner = BBoxCenterToCorner(axis=-1)
        self._foreground_iou_thresh = foreground_iou_thresh
        self._background_iou_thresh = background_iou_thresh

    def forward(self, anchors, gt_boxes):
        F = mx.nd
        anchors = self._centertocorner(anchors)
        ious = F.transpose(F.contrib.box_iou(lhs=anchors[0], rhs=gt_boxes, format='corner'), axes=(1, 0, 2))
        index = F.argmax(ious, axis=-1)  # 하나 뽑는다.

        matches = F.where(condition=F.pick(data=ious, index=index, axis=-1) >= self._foreground_iou_thresh,
                          x=index,
                          y=F.ones_like(index) * -1)

        if self._background_iou_thresh:
            # ignore label = -1을 처리를 위한 코드임 - 생각보다 어려웠음.
            matches_ignore1 = F.where(condition=F.pick(data=ious, index=index, axis=-1) >= self._foreground_iou_thresh,
                                      x=index + 1,
                                      y=F.ones_like(index) * -1)
            matches_ignore2 = F.where(condition=F.pick(data=ious, index=index, axis=-1) >= self._background_iou_thresh,
                                      x=index + 1,
                                      y=F.ones_like(index) * -1)
            matches_ignore = F.multiply(matches_ignore1, matches_ignore2)  # 부호가 바뀌는 부분이 ignore 부분

        # samples -> foreground : 1 / negative : -1 / ignore : 0 로 만드는 것이 목표
        marker = F.ones_like(matches)
        samples = F.where(matches >= 0, marker, marker * -1)
        if self._background_iou_thresh:
            samples = F.where(matches_ignore < 0, F.zeros_like(samples), samples)  # ignore 부분

        return anchors, matches, samples


# test
if __name__ == "__main__":
    from core import Efficient, DetectionDataset_V1
    import os

    input_size = (512, 512)
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    dataset = DetectionDataset_V1(path=os.path.join(root, 'Dataset', 'train'), input_size=(512, 512),
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225], image_normalization=True, box_normalization=False)

    num_classes = dataset.num_class
    image, label, _, _, _ = dataset[0]

    net = Efficient(version=0,
                    input_size=input_size,
                    anchor_sizes=[32, 64, 128, 256, 512],
                    anchor_size_ratios=[1, pow(2, 1 / 3), pow(2, 2 / 3)],
                    anchor_aspect_ratios=[0.5, 1, 2],
                    num_classes=num_classes,  # foreground만
                    anchor_box_offset=(0.5, 0.5),
                    anchor_box_clip=True,
                    alloc_size=[256, 256],
                    ctx=mx.cpu())

    net.hybridize(active=True, static_alloc=True, static_shape=True)

    matchsampler = MatchSampler(foreground_iou_thresh=0.5, background_iou_thresh=0.4)

    # batch 형태로 만들기
    image = image.expand_dims(axis=0)
    label = label.expand_dims(axis=0)
    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]

    _, _, anchors = net(image)
    _, matches, samples = matchsampler(anchors, gt_boxes)
    print(f"match shape : {matches.shape}")
    print(f"sample shape : {samples.shape}")
    '''
    match shape : (1, 49104)
    sample shape : (1, 49104)    
    '''
