# gluoncv에 있는 코드 참고

"""Decoder functions.
Decoders are used during testing/validation, which convert predictions back to
normal boxes, etc.
"""
import mxnet as mx
from mxnet.gluon import HybridBlock


class BoxDecoder(HybridBlock):

    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(BoxDecoder, self).__init__()
        self._stds = stds
        self._means = means

    def hybrid_forward(self, F, box_preds, anchors):
        anchor_x, anchor_y, anchor_width, anchor_height = anchors.split(axis=-1, num_outputs=4)
        norm_x, norm_y, norm_width, norm_height = F.split(box_preds, axis=-1, num_outputs=4)

        pre_box_x = F.broadcast_add(F.broadcast_mul(norm_x * self._stds[0] + self._means[0], anchor_width), anchor_x)
        pre_box_y = F.broadcast_add(F.broadcast_mul(norm_y * self._stds[1] + self._means[1], anchor_height), anchor_y)
        pre_box_w = F.broadcast_mul(F.exp(norm_width * self._stds[2] + self._means[2]), anchor_width)
        pre_box_h = F.broadcast_mul(F.exp(norm_height * self._stds[3] + self._means[3]), anchor_height)

        # center to corner
        half_w = pre_box_w / 2
        half_h = pre_box_h / 2
        xmin = pre_box_x - half_w
        ymin = pre_box_y - half_h
        xmax = pre_box_x + half_w
        ymax = pre_box_y + half_h
        return F.concat(xmin, ymin, xmax, ymax, dim=-1)


# multiclass decoder
class ClassMDecoder(HybridBlock):

    def __init__(self, num_classes=None, thresh=0.01, from_sigmoid=False):
        super(ClassMDecoder, self).__init__()
        self._num_classes = num_classes
        self._thresh = thresh
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, cls_preds):
        if not self._from_sigmoid:
            cls_preds = F.sigmoid(cls_preds, axis=-1)
        class_ids = F.argmax(cls_preds, axis=-1, keepdims=True)
        cls_preds = F.pick(cls_preds, class_ids, axis=-1, keepdims=True)

        # ex) thresh=0.01 이상인것만 뽑기
        mask = cls_preds > self._thresh
        class_ids = F.where(mask, class_ids, F.ones_like(class_ids) * -1)
        scores = F.where(mask, cls_preds, F.zeros_like(cls_preds))
        return class_ids, scores


# multiclass per decoder
class ClassMPDecoder(HybridBlock):

    def __init__(self, num_classes=None, thresh=0.05, from_sigmoid=False):
        super(ClassMPDecoder, self).__init__()
        self._num_classes = num_classes
        self._thresh = thresh
        self._from_sigmoid = from_sigmoid

    def hybrid_forward(self, F, cls_preds):

        if not self._from_sigmoid:
            cls_preds = F.sigmoid(cls_preds)
        # batch x all feature number x foreground class(N) -> batch x all feature number x 1 - 클래스별로 쪼개기
        template = F.zeros_like(cls_preds.slice_axis(axis=-1, begin=0, end=1))  # batch x all feature number x 1
        class_ids = []
        # batch x all feature number x 1 당 번호 0부터 부여하기
        for i in range(self._num_classes):
            class_ids.append(template + i)  # batch x all feature number x 1

        # batch x all feature number x foreground class 형태로 만들기
        class_ids = F.concat(*class_ids, dim=-1)

        # ex) thresh=0.05 이상인것만 뽑기
        mask = cls_preds > self._thresh
        class_ids = F.where(mask, class_ids, F.ones_like(class_ids) * -1)
        scores = F.where(mask, cls_preds, F.zeros_like(cls_preds))
        return class_ids, scores


''' 
    RetinaNet 논문을 읽고 구현해 봄
    모든 박스를 decoding 할 필요는 없다. 
'''


class BoxDecodeLimit(HybridBlock):
    '''
    Parameters
    ----------
    decode_number : int / -1 : all
    '''

    def __init__(self, decode_number=1000):
        super(BoxDecodeLimit, self).__init__()
        self._decode_number = decode_number

    def hybrid_forward(self, F, box_preds, anchors, class_ids, class_scores):

        if self._decode_number > 0:
            cls_scores_argmax = F.argmax(class_scores, axis=-1)  # (batch, all feature number)
            cls_scores_argsort = F.argsort(cls_scores_argmax, axis=1, is_ascend=False)
            cls_scores_argsort = F.slice_axis(cls_scores_argsort, axis=-1, begin=0,
                                              end=self._decode_number)  # (batch, self._decode_number)
            class_ids = F.take(class_ids, cls_scores_argsort, axis=1)[0]
            class_scores = F.take(class_scores, cls_scores_argsort, axis=1)[0]
            box_preds = F.take(box_preds, cls_scores_argsort, axis=1)[0]
            anchors = F.take(anchors, cls_scores_argsort, axis=1)[0]
            return class_ids, class_scores, box_preds, anchors
        else:
            return class_ids, class_scores, box_preds, anchors


# test
if __name__ == "__main__":
    from core import RetinaNet, RetinaTrainTransform, DetectionDataset
    import os

    input_size = (512, 512)
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    transform = RetinaTrainTransform(input_size[0], input_size[1], make_target=False)
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), transform=transform)
    num_classes = dataset.num_class

    image, label, _, _, _ = dataset[0]
    label = mx.nd.array(label)

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

    # batch 형태로 만들기
    image = image.expand_dims(axis=0)
    cls_preds, box_preds, anchors = net(image)

    boxdecoder = BoxDecoder(stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.))
    # classdecoder = ClassMDecoder(num_classes=num_classes, thresh=0.01, from_sigmoid=False)
    classdecoder = ClassMPDecoder(num_classes=num_classes, thresh=0.05, from_sigmoid=False)
    box_predictions = boxdecoder(box_preds, anchors)
    class_ids, class_scores = classdecoder(cls_preds)

    print(f"class id shape : {class_ids.shape}")
    print(f"class scores shape : {class_scores.shape}")
    print(f"box predictions shape : {box_predictions.shape}")
    '''
    class id shape : (1, 49104, 5)
    class scores shape : (1, 49104, 5)
    box predictions shape : (1, 49104, 4)
    '''
