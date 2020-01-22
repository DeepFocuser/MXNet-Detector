import mxnet as mx
import numpy as np
from mxnet.gluon import Block

from core.utils.dataprocessing.targetFunction.matching import Matcher


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


class Encoderfix(Block):

    def __init__(self, ignore_threshold=0.5):
        super(Encoderfix, self).__init__()
        self._cornertocenter = BBoxCornerToCenter(axis=-1)
        self._ignore_threshold = ignore_threshold

    def forward(self, matches, ious, outputs, anchors, gt_boxes, gt_ids, input_size):

        F = mx.nd
        in_height = input_size[0]
        in_width = input_size[1]
        feature_size = []
        anchor_size = []

        for out, anchor in zip(outputs, anchors):
            _, h, w, _ = out.shape
            _, _, a, _ = anchor.shape
            feature_size.append([h, w])
            anchor_size.append(a)
        all_anchors = F.concat(*[anchor.reshape(-1, 2) for anchor in anchors], dim=0)
        num_anchors = np.cumsum([anchor_size for anchor_size in anchor_size])  # ex) (3, 6, 9)
        num_offsets = np.cumsum([np.prod(feature) for feature in feature_size])  # ex) (338, 1690, 3549)
        offsets = [0] + num_offsets.tolist()

        # target 공간 만들어 놓기
        xcyc_targets = F.zeros(shape=(gt_boxes.shape[0],
                                      num_offsets[-1],
                                      num_anchors[-1], 2),
                               ctx=gt_boxes.context,
                               dtype=gt_boxes.dtype)  # (batch, 3549, 9, 2)가 기본 요소
        wh_targets = F.zeros_like(xcyc_targets)
        weights = F.zeros_like(xcyc_targets)
        objectness = F.zeros_like(xcyc_targets.split(axis=-1, num_outputs=2)[0])
        class_targets = F.zeros_like(objectness)

        gtx, gty, gtw, gth = self._cornertocenter(gt_boxes)
        np_gtx, np_gty, np_gtw, np_gth = [x.asnumpy() for x in [gtx, gty, gtw, gth]]
        np_anchors = all_anchors.asnumpy()
        np_gt_ids = gt_ids.asnumpy()
        np_gt_ids = np_gt_ids.astype(int)

        # 가장 큰것에 할당하고, target anchor 비교해서 0.5 이상인것들 무시하기
        batch, anchorN, objectN = ious.shape

        for b in range(batch):
            for a in range(anchorN):
                for o in range(objectN):
                    nlayer = np.where(num_anchors > a)[0][0]
                    out_height = outputs[nlayer].shape[1]
                    out_width = outputs[nlayer].shape[2]

                    gtx, gty, gtw, gth = (np_gtx[b, o, 0], np_gty[b, o, 0],
                                          np_gtw[b, o, 0], np_gth[b, o, 0])

                    ''' 
                        matching 단계에서 image만 들어온 데이터들도 matching이 되기때문에 아래와 같이 걸러줘야 한다.
                        image만 들어온 데이터 or padding 된것들은 noobject이다. 
                    '''
                    if gtx == -1.0 and gty == -1.0 and gtw == 0.0 and gth == 0.0:
                        continue
                    # compute the location of the gt centers
                    loc_x = int(gtx / in_width * out_width)
                    loc_y = int(gty / in_height * out_height)
                    # write back to targets
                    index = offsets[nlayer] + loc_y * out_width + loc_x
                    if a == matches[b, o]:  # 최대인 값은 제외
                        xcyc_targets[b, index, a, 0] = gtx / in_width * out_width - loc_x
                        xcyc_targets[b, index, a, 1] = gty / in_height * out_height - loc_y
                        '''
                        if gtx == -1.0 and gty == -1.0 and gtw == 0.0 and gth == 0.0: 
                            continue
                        에서 처리를 해주었으나, 그래도 한번 더 대비 해놓자.
                        '''
                        wh_targets[b, index, a, 0] = np.log(
                            max(gtw, 1) / np_anchors[a, 0])  # max(gtw,1)? gtw, gth가 0일경우가 있다.
                        wh_targets[b, index, a, 1] = np.log(max(gth, 1) / np_anchors[a, 1])
                        weights[b, index, a, :] = 2.0 - gtw * gth / in_width / in_height
                        objectness[b, index, a, 0] = 1
                        class_targets[b, index, a, 0] = np_gt_ids[b, o, 0]
                        continue
                    if ious[b, a, o] >= self._ignore_threshold:
                        objectness[b, index, a, 0] = -1

        xcyc_targets = self._slice(xcyc_targets, num_anchors, num_offsets)
        wh_targets = self._slice(wh_targets, num_anchors, num_offsets)
        weights = self._slice(weights, num_anchors, num_offsets)
        objectness = self._slice(objectness, num_anchors, num_offsets)
        class_targets = self._slice(class_targets, num_anchors, num_offsets)
        class_targets = F.squeeze(class_targets, axis=-1)

        # threshold 바꿔가며 개수 세어보기
        # print((objectness == 1).sum().asscalar().astype(int))
        # print((objectness == 0).sum().asscalar().astype(int))
        # print((objectness == -1).sum().asscalar().astype(int))

        return xcyc_targets, wh_targets, objectness, class_targets, weights

    def _slice(self, x, num_anchors, num_offsets):
        F = mx.nd
        anchors = [0] + num_anchors.tolist()
        offsets = [0] + num_offsets.tolist()
        ret = []
        for i in range(len(num_anchors)):
            y = x[:, offsets[i]:offsets[i + 1], anchors[i]:anchors[i + 1], :]
            ret.append(y.reshape(0, -3, -1))
        return F.concat(*ret, dim=1)


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

    matcher = Matcher()
    encoder = Encoderfix(ignore_threshold=0.5)

    # batch 형태로 만들기
    image = image.expand_dims(axis=0)
    label = label.expand_dims(axis=0)

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    output1, output2, output3, anchor1, anchor2, anchor3, _, _, _, _, _, _ = net(image)

    matches, ious = matcher([anchor1, anchor2, anchor3], gt_boxes)
    xcyc_targets, wh_targets, objectness, class_targets, weights = encoder(matches, ious, [output1, output2, output3],
                                                                           [anchor1, anchor2, anchor3], gt_boxes,
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
