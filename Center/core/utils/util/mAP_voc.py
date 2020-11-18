"""
Pascal VOC Detection evaluation.
< reference >
https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
"""
import os
from collections import defaultdict

import numpy as np
import plotly.graph_objs as go

COLOR = defaultdict(lambda: (0, 0, 0))
COLOR[0] = (134, 229, 127)
COLOR[1] = (255, 0, 0)  # 빨
COLOR[2] = (255, 130, 36)  # 주
COLOR[3] = (250, 237, 128)  # 노
COLOR[4] = (0, 255, 0)  # 초
COLOR[5] = (0, 0, 255)  # 파
COLOR[6] = (3, 0, 102)  # 남
COLOR[7] = (128, 65, 217)  # 보
COLOR[8] = (61, 183, 204)
COLOR[9] = (217, 65, 197)
COLOR[10] = (250, 236, 197)


def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.
    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.
    """
    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")
    # None 넣으면 axis 하나 더 생김
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # (N,1,2) , (M,2) -> (N,M,2)
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])  # (N,1,2) , (M,2) -> (N,M,2)
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)  # (N,M) * (N,M)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)  # (N,)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)  # (M,)
    return area_i / (area_a[:, None] + area_b - area_i)  # (N,M) / (N,1) + (M,) - (N,M)


class Voc_base_PR(object):
    """
    Parameters:

    iou_thresh : true positive 를 위한 IOU(intersection Over Union) 임계값
    class_names : 각 클래스 이름
    """

    def __init__(self, iou_thresh=0.5, class_names=None, offset=0):
        super(Voc_base_PR, self).__init__()

        if not isinstance(class_names, (tuple, list)):
            raise TypeError  # list or tuple 이어야 함.
        else:
            for name in class_names:
                if not isinstance(name, str):
                    raise TypeError  # 문자열 이어어 함

        self._class_names = class_names
        self._class_number = len(class_names)
        self._iou_thresh = iou_thresh
        self._offset = offset
        self.reset()

    def update(self, pred_bboxes=None,
               pred_labels=None,
               pred_scores=None,
               gt_boxes=None,
               gt_labels=None):

        """
        최종 network 출력 결과물(depcoder + nms)
        pred_boxes : mx.nd.ndarray / shape : (batch size , object number N, 4)
        (N is the number of bbox)
        pred_labels : mx.nd.ndarray / shape : (batch size , object number)
        pred_scores : mx.nd.ndarray / shape : (batch size , object number)

        Ground truth를 normalization한 결과물
        gt_boxes : mx.nd.ndarray / shape : (batch size , object number M, 4
        (M is the number of ground-truths)
        gt_labels : mx.nd.ndarray /   (batch size , object number M)
        """

        pred_bboxes = pred_bboxes.asnumpy()
        pred_labels = pred_labels.asnumpy()
        pred_scores = pred_scores.asnumpy()
        gt_boxes = gt_boxes.asnumpy()
        gt_labels = gt_labels.asnumpy()

        for pred_bbox, pred_label, pred_score, gt_box, gt_label in zip(
                pred_bboxes, pred_labels, pred_scores, gt_boxes, gt_labels):

            # pred_label 이 0번 부터를 정상으로 본다.
            valid_pred = np.where(pred_label.ravel() >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :]
            pred_label = pred_label.ravel()[valid_pred].astype(int)
            pred_score = pred_score.ravel()[valid_pred]

            # gt_label 이 0번 부터를 정상으로 본다.
            valid_gt = np.where(gt_label.ravel() >= 0)[0]
            gt_box = gt_box[valid_gt, :]
            gt_label = gt_label.ravel()[valid_gt].astype(int)

            # np.unique(x) : 배열 내 중복된 원소 제거 후 유일한 원소를 정렬하여 반환 - class별로 계산하기 위함.
            for i in np.unique(np.concatenate([pred_label, gt_label])):  # class 한개씩

                pred_mask_i = np.where(pred_label == i)[0]  # i class predicion 것만 보겠다.
                pred_box_i = pred_bbox[pred_mask_i, :]
                pred_score_i = pred_score[pred_mask_i]

                order = pred_score_i.argsort()[::-1]  # score에 따라 내림 차순 정렬하겠다.
                pred_box_i = pred_box_i[order]
                pred_score_i = pred_score_i[order]

                gt_mask_i = np.where(gt_label == i)[0]  # i class ground truths 것만 보겠다.
                gt_box_i = gt_box[gt_mask_i, :]

                self._positive_number[i] += len(gt_mask_i)  # 클래스별 True 구하기

                if len(pred_score_i) == 0 and len(pred_box_i) == 0:
                    continue

                self._score[i].extend(pred_score_i)

                '''
                아래의 3줄이 은근히 중요한데, 그 이유는
                데이터에 정답이 없는데도, 네트워크가
                정답으로 분류해 버릴수도 있기 때문이다.(원숭이와 침팬치를 구분하기 힘들어하는 것을 예로 들을수 있겠다.)
                이럴 경우 무조건 false_positive 이다.
                아래의 3줄의 코드를 넣는게, 더 정확한 AP를 구하는 것이다.

                사실 아래의 3줄이 동작하기 전에, 
                if len(pred_score_i) == 0 and len(pred_box_i) == 0:
                    continue
                에서 다 걸러져 버리는 것이 좋을 듯 하다.
                '''

                if len(gt_box_i) == 0:
                    self._match[i].extend((0,) * pred_box_i.shape[0])
                    continue

                pred_box_i = pred_box_i.copy()
                pred_box_i[:, 2:] += self._offset
                gt_box_i = gt_box_i.copy()
                gt_box_i[:, 2:] += self._offset

                iou = bbox_iou(pred_box_i, gt_box_i)  # (N,M)
                gt_index = iou.argmax(axis=-1)  # (N,) 범위 0 ~ M - 1

                # self._iou_thresh 보다 작은 값들은 -1로 채우기
                gt_index[iou.max(axis=1) < self._iou_thresh] = -1  # (N,)

                # matching 하는 부분
                selection = np.zeros(gt_box_i.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:  # true_positive
                        if not selection[gt_idx]:  # gt_box_i 개수 만큼만 더하겠다.
                            self._match[i].append(1)
                        else:
                            self._match[i].append(0)
                        selection[gt_idx] = True
                    else:  # false_positive
                        self._match[i].append(0)

    def get_PR_list(self):

        class_name = [f"{self._class_names[i]}" for i in range(self._class_number)]
        precision = list([None]) * self._class_number
        recall = list([None]) * self._class_number
        true_positive = list([None]) * self._class_number
        false_positive = list([None]) * self._class_number
        threshold = list([None]) * self._class_number

        for i in range(self._class_number):
            score = np.array(self._score[i])
            match = np.array(self._match[i], dtype=np.int32)

            # 내림차순 정렬
            order = score.argsort()[::-1]
            threshold[i] = np.sort(score)[::-1]
            match = match[order]

            # 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수.
            true_positive[i] = np.cumsum(match == 1)
            false_positive[i] = np.cumsum(match == 0)

            # self._positive_number[l] > 0 인 경우, Precision, Recall 계산하여 반환
            # self._positive_number[l] <= 0 인 경우, None 반환
            if self._positive_number[i] > 0:
                recall[i] = true_positive[i] / self._positive_number[i]
                # fp + tp == 0인 경우, precision 0,
                precision[i] = np.nan_to_num(np.divide(true_positive[i], (false_positive[i] + true_positive[i])))

        return class_name, precision, recall, true_positive, false_positive, threshold

    def get_PR_curve(self, name=None, precision=None, recall=None, threshold=None, AP=None, mAP=None, root="",
                     folder_name="test_graph", epoch=None, auto_open=False):

        ceil_position = 2

        graph_save_path = os.path.join(root, folder_name)
        if not os.path.exists(graph_save_path):
            os.makedirs(graph_save_path)

        AP = np.nan_to_num(AP)
        mAP = np.nan_to_num(mAP)

        order = AP.argsort()[::-1]  # Average Precision에 따른 내림 차순 정렬

        name = np.array(name)[order]
        precision = np.array(precision, dtype=np.object)[order]
        recall = np.array(recall, dtype=np.object)[order]
        threshold = np.array(threshold, dtype=np.object)[order]
        AP = np.around(AP[order] * 100, ceil_position)

        fig = go.Figure()
        index = 0
        for n, p, r, t, ap in zip(name, precision, recall, threshold, AP):

            # t가 비어있는 경우
            if not t.tolist():
                p = []
                r = []
                t = []
                p.append(0.0)
                r.append(0.0)
                t.append(0.0)

            # t는 있는데, p,r이 None인 경우
            try:
                p = p.tolist()
                r = r.tolist()
            except Exception as E:
                p = [0]*len(t)
                r = [0]*len(t)

            fig.add_trace(go.Scatter(
                x=r,
                y=p,
                mode="lines",
                name=f"{n}({ap}%)",
                text=[f"score : {round(score * 100, ceil_position)}" for score in t],
                marker=dict(
                    color=f'rgb{COLOR[index]}'),
            ))
            index+=1

        fig.update_layout(title=f'Mean Average Precision : {round(mAP * 100, ceil_position)}%',
                          xaxis_title='Recall',
                          yaxis_title='Precision',
                          xaxis=dict(range=[0, 1]),
                          yaxis=dict(range=[0, 1]),
                          legend=dict(
                              y=0.5,
                              font=dict(
                                  size=20
                              )
                          )
                          )

        if isinstance(epoch, int):
            fig.write_html(file=os.path.join(graph_save_path, f'{self.__repr__()}mAP_{epoch}epoch_line.html'),
                           auto_open=auto_open)
        else:
            fig.write_html(file=os.path.join(graph_save_path, f'{self.__repr__()}mAP_line.html'),
                           auto_open=auto_open)


        # vertical bar
        fig = go.Figure(data=[go.Bar(
            x=[n for n in name],
            y=[ap for ap in AP],
            text=[f'{str(ap)}%' for ap in AP],
            textposition = 'auto',
            marker=dict(
                color=[f'rgb{COLOR[i]}' for i in range(len(name))]),
        )])

        fig.update_layout(title=f'Mean Average Precision : {round(mAP * 100, ceil_position)}%',
                          xaxis_title='class',
                          yaxis_title='AP',
                          yaxis=dict(range=[0, 100]))

        if isinstance(epoch, int):
            fig.write_html(file=os.path.join(graph_save_path, f'{self.__repr__()}mAP_{epoch}epoch_vbar.html'),
                           auto_open=auto_open)
        else:
            fig.write_html(file=os.path.join(graph_save_path, f'{self.__repr__()}mAP_vbar.html'),
                           auto_open=auto_open)

    def reset(self):

        self._positive_number = defaultdict(int)  # 각 클래스별 진짜 양성 개수 - True per class
        self._score = defaultdict(list)  # 각 클래스별 점수
        self._match = defaultdict(list)  # 각 클래스별 matching 개수 - for true positive , False positive per class


'''
an average for th 11-point interpolated AP is calculated.
11-point interpolation
'''


class Voc_2007_AP(Voc_base_PR):

    def __init__(self, point=11, *args, **kwargs):
        super(Voc_2007_AP, self).__init__(*args, **kwargs)
        assert isinstance(point, (int, float)), "point's data type must be 'real' or 'integer'"
        self._point = point

    def __repr__(self):
        return "voc2007"

    '''
    get_AP 
        Average Precision
    '''

    def get_AP(self, name, precision, recall):

        """
        Parameters:

        recall : numpy.array list
            cumulated recall
        precision : numpy.array list
            cumulated precision
        Return:
            Average Precision per class
        """
        if precision is None or recall is None:
            return name, np.nan

        AP = 0.
        for t in np.arange(0.0, 1.1, 1 / (self._point - 1)):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            AP += p / self._point
        return name, AP


'''
voc2010-2012 samples the curve at all unique recall values,
whenever the maximum precision value drop 
AUC(Area Under Curve AUC)
Interpolating all points
'''


class Voc_2010_AP(Voc_base_PR):

    def __init__(self, *args, **kwargs):
        super(Voc_2010_AP, self).__init__(*args, **kwargs)

    '''
    get_AP 
        Average Precision
    '''

    def __repr__(self):
        return "voc2010"

    def get_AP(self, name, precision, recall):
        """
        Parameters:

        recall : numpy.array list
            cumulated recall
        precision : numpy.array list
            cumulated precision
        Return:
            Average Precision per class
        """
        if precision is None or recall is None:
            return name, np.nan

        # precision, recall 양 끝에, start, end 값 채워주기
        precision = np.concatenate(([0.], precision, [0.]))
        recall = np.concatenate(([0.], recall, [1.]))

        '''
        voc2010-2012 samples the curve at all 'unique' recall values,
        (unique라는 말은 중복되는 부분 제거)
        whenever the maximum precision value drop

        최댓값이 떨어지는 지점마다 샘플링해서 보간하겠다.
        '''
        for i in range(0, precision.size - 1, 1):
            precision[i] = np.maximum(precision[i], precision[i + 1])

        '''
        recall(x)축을 쭉 보면서, 변화가 있는 부분 sampling
        왜 이렇게 하나?
        recall은 같은 값을 가지는데, precision은 변하는 부분이 있을 것이다.
        이것을 고려하지 않고 면적을 계산하기가 힘들다. 

        recall의 변화가 중복되는 부분을 제거한다.(precision은 이미 위에서 처리 했으므로 바로 적용 가능)
        '''
        index = np.where(recall[1:] != recall[:-1])[0]
        AP = np.sum((recall[index + 1] - recall[index]) * precision[index + 1])
        return name, AP


if __name__ == "__main__":
    import random
    from core import CenterNet, DetectionDataset, CenterValidTransform
    from core import Prediction
    from collections import OrderedDict
    import mxnet as mx

    input_size = (512, 512)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = CenterValidTransform(input_size, mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225), make_target=False)
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), transform=transform)
    num_classes = dataset.num_class
    name_classes = dataset.classes
    length = len(dataset)
    image, label, _, _, _ = dataset[random.randint(0, length - 1)]

    net = CenterNet(base=18,
                    heads=OrderedDict([
                        ('heatmap', {'num_output': num_classes, 'bias': -2.19}),
                        ('offset', {'num_output': 2}),
                        ('wh', {'num_output': 2})
                    ]),
                    head_conv_channel=64, pretrained=False,
                    root=os.path.join(root, "modelparam"),
                    use_dcnv2=False,
                    ctx=mx.cpu())
    net.hybridize(active=True, static_alloc=True, static_shape=True)

    prediction = Prediction(topk=100, scale=4)
    precision_recall_2007 = Voc_2007_AP(iou_thresh=0.5, class_names=name_classes)
    precision_recall_2010 = Voc_2010_AP(iou_thresh=0.5, class_names=name_classes)

    # batch 형태로 만들기
    data = image.expand_dims(axis=0)
    label = np.expand_dims(label, axis=0)
    label = mx.nd.array(label)

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]

    heatmap_pred, offset_pred, wh_pred = net(data)
    ids, scores, bboxes = prediction(heatmap_pred, offset_pred, wh_pred)

    precision_recall_2007.update(pred_bboxes=bboxes,
                                 pred_labels=ids,
                                 pred_scores=scores,
                                 gt_boxes=gt_boxes * 4,
                                 gt_labels=gt_ids)
    precision_recall_2010.update(pred_bboxes=bboxes,
                                 pred_labels=ids,
                                 pred_scores=scores,
                                 gt_boxes=gt_boxes * 4,
                                 gt_labels=gt_ids)

    AP_appender = []
    round_position = 2
    class_name, precision, recall, true_positive, false_positive, threshold = precision_recall_2007.get_PR_list()
    print("<<< voc2007 >>>")
    for i, c, p, r in zip(range(len(recall)), class_name, precision, recall):
        name, AP = precision_recall_2007.get_AP(c, p, r)
        print(f"class {i}'s {name} AP : {round(AP * 100, round_position)} %")
        AP_appender.append(AP)
    mAP_result = np.mean(AP_appender)

    print(f"mAP : {round(mAP_result * 100, round_position)} %")
    precision_recall_2007.get_PR_curve(name=class_name,
                                       precision=precision,
                                       recall=recall,
                                       threshold=threshold,
                                       AP=AP_appender, mAP=mAP_result, root=root)
    print("\n")

    AP_appender = []
    class_name, precision, recall, true_positive, false_positive, threshold = precision_recall_2010.get_PR_list()

    print("<<< voc2010 >>>")
    for i, c, p, r in zip(range(len(recall)), class_name, precision, recall):
        name, AP = precision_recall_2010.get_AP(c, p, r)
        print(f"class {i}'s {name} AP : {round(AP * 100, round_position)} %")
        AP_appender.append(AP)

    # 각 클래스별 ap 확률 막대로 표시하고 맨 위에 mAP 표시해주기
    mAP_result = np.mean(AP_appender)
    print(f"mAP : {round(mAP_result * 100, round_position)} %")

    precision_recall_2010.get_PR_curve(name=class_name,
                                       precision=precision,
                                       recall=recall,
                                       threshold=threshold,
                                       AP=AP_appender, mAP=mAP_result, root=root)
