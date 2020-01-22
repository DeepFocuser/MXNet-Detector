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


class Matcher(Block):

    def __init__(self):
        super(Matcher, self).__init__()
        self._cornertocenter = BBoxCornerToCenter(axis=-1)
        self._centertocorner = BBoxCenterToCorner(axis=-1)

    def forward(self, anchors, gt_boxes):
        F = mx.nd

        '''
        gt_box : 중심이 0인 공간으로 gt를 mapping하는 방법
        -> grid cell 기반이라서 이러한 방법으로 matching 가능
         anchor와 gt의 중심점은 공유된다.
        '''
        gtx, gty, gtw, gth = self._cornertocenter(gt_boxes)  # 1. gt를 corner -> center로 변경하기
        shift_gt_boxes = mx.nd.concat(-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth, dim=-1)  # 중심점이 0,0인 corner로 바꾸기
        '''
        anchor는 width, height를 알고 있으니 중심점이 0, 0 을 가리키도록 한다. 
        '''
        all_anchors = mx.nd.concat(*[a.reshape(-1, 2) for a in anchors], dim=0)
        anchor_boxes = mx.nd.concat(0 * all_anchors, all_anchors, dim=-1)  # zero center anchors / (9, 4)
        anchor_boxes = self._centertocorner(anchor_boxes)

        # anchor_boxes : (9, 4) / gt_boxes : (Batch, N, 4) -> (9, Batch, N) -> (Batch, 9, N)
        ious = F.transpose(F.contrib.box_iou(anchor_boxes, shift_gt_boxes), axes=(1, 0, 2))
        '''
        matching process에 대한 정확한 이해가 필요하다. SSD, Retinanet vs Yolo
        '''
        matches = ious.argmax(axis=1).asnumpy()  # (Batch, N) / 가장 큰것 하나만 뽑는다.
        matches = matches.astype(int)
        ious = ious.asnumpy()

        return matches, ious


# test
if __name__ == "__main__":
    from core import Yolov3, DetectionDataset_V1
    import os

    input_size = (416, 416)
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    dataset = DetectionDataset_V1(path=os.path.join(root, 'Dataset', 'train'), input_size=(512, 512),
                                  mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225], image_normalization=True, box_normalization=False)

    num_classes = dataset.num_class
    image, label, _, _, _ = dataset[0]

    net = Yolov3(Darknetlayer=53,
                 input_size=input_size,
                 anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                          "middle": [(30, 61), (62, 45), (59, 119)],
                          "deep": [(116, 90), (156, 198), (373, 326)]},
                 num_classes=num_classes,  # foreground만
                 pretrained=False,
                 pretrained_path=os.path.join(root, "modelparam"),
                 ctx=mx.cpu())
    net.hybridize(active=True, static_alloc=True, static_shape=True)

    centertocorner = BBoxCenterToCorner(axis=-1)
    cornertocenter = BBoxCornerToCenter(axis=-1)
    matcher = Matcher()

    # batch 형태로 만들기
    image = image.expand_dims(axis=0)
    label = label.expand_dims(axis=0)
    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]

    _, _, _, anchor1, anchor2, anchor3, _, _, _, _, _, _ = net(image)
    matches, ious = matcher([anchor1, anchor2, anchor3], gt_boxes)
    print(f"match shape : {matches.shape}")
    print(f"iou shape : {ious.shape}")
    '''
    match shape : (1, 1)
    iou shape : (1, 9, 1)
    '''
