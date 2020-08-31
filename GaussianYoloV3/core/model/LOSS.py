# Gluoncv Loss에서 수정함.

import math

from mxnet.gluon import HybridBlock

'''
https://arxiv.org/pdf/1904.04620.pdf
NLL(negative log likelihood loss) LOSS
objectness and class 관련 loss는 YoloV3 그대로 이다.
'''


class GaussianYolov3Loss(HybridBlock):

    def __init__(self, sparse_label=True,
                 from_sigmoid=False,
                 batch_axis=None,
                 num_classes=5,
                 reduction="sum",
                 exclude=False,
                 epsilon=1e-9):

        super(GaussianYolov3Loss, self).__init__()
        self._sparse_label = sparse_label
        self._from_sigmoid = from_sigmoid
        self._batch_axis = batch_axis
        self._num_classes = num_classes
        self._reduction = reduction.upper()
        self._exclude = exclude
        self._num_pred = 9 + num_classes

        self._sigmoid_ce = SigmoidBinaryCrossEntropyLoss(from_sigmoid=from_sigmoid,
                                                         batch_axis=batch_axis,
                                                         reduction=reduction,
                                                         exclude=exclude)
        self._NLLloss = NegativeLogLikelihoodLoss(batch_axis=batch_axis,
                                                  reduction=reduction,
                                                  exclude=exclude, epsilon=epsilon)

    def hybrid_forward(self, F, output1, output2, output3, xcyc_target, wh_target, objectness, class_target, weights):

        # 1. prediction 쪼개기
        pred = F.concat(*[out.reshape(0, -1, self._num_pred) for out in [output1, output2, output3]], dim=1)
        xc_mean_pred = F.slice_axis(data=pred, axis=-1, begin=0, end=1)
        xc_var_pred = F.slice_axis(data=pred, axis=-1, begin=1, end=2)
        yc_mean_pred = F.slice_axis(data=pred, axis=-1, begin=2, end=3)
        yc_var_pred = F.slice_axis(data=pred, axis=-1, begin=3, end=4)

        w_mean_pred = F.slice_axis(data=pred, axis=-1, begin=4, end=5)
        w_var_pred = F.slice_axis(data=pred, axis=-1, begin=5, end=6)
        h_mean_pred = F.slice_axis(data=pred, axis=-1, begin=6, end=7)
        h_var_pred = F.slice_axis(data=pred, axis=-1, begin=7, end=8)
        objectness_pred = F.slice_axis(data=pred, axis=-1, begin=8, end=9)
        class_pred = F.slice_axis(data=pred, axis=-1, begin=9, end=None)

        # 2. target 쪼개기
        xc_target = F.slice_axis(data=xcyc_target, axis=-1, begin=0, end=1)
        yc_target = F.slice_axis(data=xcyc_target, axis=-1, begin=1, end=None)
        w_target = F.slice_axis(data=wh_target, axis=-1, begin=0, end=1)
        h_target = F.slice_axis(data=wh_target, axis=-1, begin=1, end=None)

        # 2. loss 구하기
        object_mask = objectness == 1
        noobject_mask = objectness == 0

        # coordinates loss
        if not self._from_sigmoid:
            xc_mean_pred = F.sigmoid(xc_mean_pred)
            xc_var_pred = F.sigmoid(xc_var_pred)
            yc_mean_pred = F.sigmoid(yc_mean_pred)
            yc_var_pred = F.sigmoid(yc_var_pred)
            w_var_pred = F.sigmoid(w_var_pred)
            h_var_pred = F.sigmoid(h_var_pred)

        xc_loss = self._NLLloss(xc_mean_pred, xc_var_pred, xc_target, object_mask * weights * 0.5)
        yc_loss = self._NLLloss(yc_mean_pred, yc_var_pred, yc_target, object_mask * weights * 0.5)
        w_loss = self._NLLloss(w_mean_pred, w_var_pred, w_target, object_mask * weights * 0.5)
        h_loss = self._NLLloss(h_mean_pred, h_var_pred, h_target, object_mask * weights * 0.5)

        xcyc_loss = xc_loss + yc_loss
        wh_loss = w_loss + h_loss

        # object loss + noboject loss
        obj_loss = self._sigmoid_ce(objectness_pred, objectness, object_mask)
        noobj_loss = self._sigmoid_ce(objectness_pred, objectness, noobject_mask)
        object_loss = F.add(noobj_loss, obj_loss)

        if self._sparse_label:
            class_target = F.one_hot(class_target, self._num_classes)
        # class loss
        class_loss = self._sigmoid_ce(class_pred, class_target, object_mask)

        return xcyc_loss, wh_loss, object_loss, class_loss


class NegativeLogLikelihoodLoss(HybridBlock):

    def __init__(self, batch_axis=0, reduction="sum", exclude=False, epsilon=1e-9):
        super(NegativeLogLikelihoodLoss, self).__init__()

        self._batch_axis = batch_axis
        self._reduction = reduction.upper()
        self._exclude = exclude
        self._epsilon = epsilon

    def hybrid_forward(self, F, u, sigma, x, sample_weight=None):

        '''
            Negative LogLikelihood Loss
            Normal Distribution
            https://en.wikipedia.org/wiki/Normal_distribution
        '''
        first = F.broadcast_div(F.ones_like(u), F.sqrt(2 * math.pi * (sigma + 0.3 +self._epsilon)))
        second = F.exp(-F.broadcast_div(F.square(x - u), 2.0 * (sigma + self._epsilon)))
        loss = -F.log(F.broadcast_mul(first, second))

        if sample_weight is not None:
            loss = F.broadcast_mul(loss, sample_weight)
        if self._reduction == "SUM":
            return F.sum(loss, axis=self._batch_axis, exclude=self._exclude)
        elif self._reduction == "MEAN":
            return F.mean(loss, axis=self._batch_axis, exclude=self._exclude)
        else:
            raise NotImplementedError


class SigmoidBinaryCrossEntropyLoss(HybridBlock):

    def __init__(self, from_sigmoid=False, batch_axis=0, reduction="sum", exclude=False):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._batch_axis = batch_axis
        self._reduction = reduction.upper()
        self._exclude = exclude

    def hybrid_forward(self, F, pred, label, sample_weight=None, pos_weight=None):

        if not self._from_sigmoid:
            if pos_weight is None:
                # We use the stable formula: max(x, 0) - x * z + log(1 + exp(-abs(x)))
                loss = F.relu(pred) - pred * label + \
                       F.Activation(-F.abs(pred), act_type='softrelu')
            else:
                # We use the stable formula: x - x * z + (1 + z * pos_weight - z) * \
                #    (log(1 + exp(-abs(x))) + max(-x, 0))
                log_weight = 1 + F.broadcast_mul(pos_weight - 1, label)
                loss = pred - pred * label + log_weight * \
                       (F.Activation(-F.abs(pred), act_type='softrelu') + F.relu(-pred))
        else:
            eps = 1e-12
            if pos_weight is None:
                loss = -(F.log(pred + eps) * label
                         + F.log(1. - pred + eps) * (1. - label))
            else:
                loss = -(F.broadcast_mul(F.log(pred + eps) * label, pos_weight)
                         + F.log(1. - pred + eps) * (1. - label))
        if sample_weight is not None:
            loss = F.broadcast_mul(loss, sample_weight)

        if self._reduction == "SUM":
            return F.sum(loss, axis=self._batch_axis, exclude=self._exclude)
        elif self._reduction == "MEAN":
            return F.mean(loss, axis=self._batch_axis, exclude=self._exclude)
        else:
            raise NotImplementedError
