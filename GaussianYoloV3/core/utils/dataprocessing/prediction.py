import mxnet as mx
from mxnet.gluon import HybridBlock

from core.utils.dataprocessing.predictFunction.decoder import Decoder

'''
    Prediction 클래스를 hybridBlock 으로 만든이유?
    net + decoder + nms =  complete object network로 만들어서 json, param 파일로 저장하기 위함
    -> 장점은? mxnet c++에서 inference 할 때, image(RGB) 넣으면 ids, scores, bboxes 가 바로 출력
'''


class Prediction(HybridBlock):

    def __init__(self,
                 from_sigmoid=False,
                 num_classes=3,
                 nms_thresh=0.5,
                 nms_topk=500,
                 except_class_thresh=0.05,
                 multiperclass=True):
        super(Prediction, self).__init__()

        self._decoder = Decoder(from_sigmoid=from_sigmoid, num_classes=num_classes, thresh=except_class_thresh,
                                multiperclass=multiperclass)
        self._nms_thresh = nms_thresh
        self._nms_topk = nms_topk

    def hybrid_forward(self, F, output1, output2, output3,
                       anchor1, anchor2, anchor3,
                       offset1, offset2, offset3,
                       stride1, stride2, stride3):

        results = []
        for out, an, off, st in zip([output1, output2, output3],
                                    [anchor1, anchor2, anchor3],
                                    [offset1, offset2, offset3],
                                    [stride1, stride2, stride3]):
            results.append(self._decoder(out, an, off, st))

        results = F.concat(*results, dim=1)
        if self._nms_thresh > 0 and self._nms_thresh < 1:
            '''
            Apply non-maximum suppression to input.
            The output will be sorted in descending order according to score. 
            Boxes with overlaps larger than overlap_thresh, 
            smaller scores and background boxes will be removed and filled with -1, 
            '''
            results = F.contrib.box_nms(
                results,
                overlap_thresh=self._nms_thresh,
                topk=self._nms_topk,
                id_index=0, score_index=1, coord_start=2,
                force_suppress=False, in_format="corner", out_format="corner")

        ids = F.slice_axis(results, axis=-1, begin=0, end=1)
        scores = F.slice_axis(results, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(results, axis=-1, begin=2, end=6)
        return ids, scores, bboxes


# test
if __name__ == "__main__":
    from core import Yolov3, YoloTrainTransform, DetectionDataset
    import os

    input_size = (416, 416)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = YoloTrainTransform(input_size[0], input_size[1], make_target=False)
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), transform=transform)
    num_classes = dataset.num_class

    image, label, _, _, _ = dataset[0]
    label = mx.nd.array(label)

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

    prediction = Prediction(
        from_sigmoid=False,
        num_classes=num_classes,
        nms_thresh=0.5,
        nms_topk=100,
        except_class_thresh=0.05,
        multiperclass=True)

    # batch 형태로 만들기
    image = image.expand_dims(axis=0)
    label = label.expand_dims(axis=0)
    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(
        image)
    ids, scores, bboxes = prediction(output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3,
                                     stride1, stride2, stride3)

    print(f"nms class id shape : {ids.shape}")
    print(f"nms class scores shape : {scores.shape}")
    print(f"nms box predictions shape : {bboxes.shape}")
    '''
    multiperclass = True 일 때,
    nms class id shape : (1, 53235, 1)
    nms class scores shape : (1, 53235, 1)
    nms box predictions shape : (1, 53235, 4)

    multiperclass = False 일 때,
    nms class id shape : (1, 10647, 1)
    nms class scores shape : (1, 10647, 1)
    nms box predictions shape : (1, 10647, 4)
    '''
