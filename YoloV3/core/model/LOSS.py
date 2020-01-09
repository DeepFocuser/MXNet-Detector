# Gluoncv Loss에서 수정함.

from mxnet.gluon import HybridBlock


# 왜 HybridBlock이 아닌 Block? output 이 list이다.
class Yolov3Loss(HybridBlock):

    def __init__(self, sparse_label = True,
                 from_sigmoid=False,
                 batch_axis=None,
                 num_classes=5,
                 reduction="sum",
                 exclude=False):

        super(Yolov3Loss, self).__init__()
        self._sparse_label = sparse_label
        self._from_sigmoid = from_sigmoid
        self._batch_axis = batch_axis
        self._num_classes = num_classes
        self._reduction = reduction.upper()
        self._exclude = exclude
        self._num_pred = 5 + num_classes

        self._sigmoid_ce = SigmoidBinaryCrossEntropyLoss(from_sigmoid=from_sigmoid,
                                                         batch_axis=batch_axis,
                                                         reduction=reduction,
                                                         exclude=exclude)
        self._l2loss = L2Loss(batch_axis=batch_axis,
                              reduction=reduction,
                              exclude=exclude)

    def hybrid_forward(self, F, output1, output2, outptut3, xcyc_target, wh_target, objectness, class_target, weights):

        #1. prediction 쪼개기
        pred = F.concat(*[out.reshape(0, -1, self._num_pred) for out in [output1, output2, outptut3]], dim=1)
        xcyc_pred = F.slice_axis(data=pred, axis=-1, begin=0, end=2)
        wh_pred = F.slice_axis(data=pred, axis=-1, begin=2, end=4)
        objectness_pred = F.slice_axis(data=pred, axis=-1, begin=4, end=5)
        class_pred = F.slice_axis(data=pred, axis=-1, begin=5, end=None)

        #2. loss 구하기
        object_mask= objectness == 1
        noobject_mask = objectness == 0

        # coordinates loss
        if not self._from_sigmoid:
            xcyc_pred = F.sigmoid(xcyc_pred)

        xcyc_loss = self._l2loss(xcyc_pred, xcyc_target, object_mask*weights)
        wh_loss =self._l2loss(wh_pred, wh_target, object_mask*weights)

        # object loss + noboject loss
        obj_loss = self._sigmoid_ce(objectness_pred, objectness, object_mask)
        noobj_loss = self._sigmoid_ce(objectness_pred, objectness, noobject_mask)
        object_loss = F.add(noobj_loss, obj_loss)

        if self._sparse_label:
            class_target = F.one_hot(class_target, self._num_classes)
        # class loss
        class_loss = self._sigmoid_ce(class_pred, class_target, object_mask)

        return xcyc_loss, wh_loss, object_loss, class_loss

class L2Loss(HybridBlock):

    def __init__(self, batch_axis=0, reduction="sum", exclude=False):
        super(L2Loss, self).__init__()

        self._batch_axis = batch_axis
        self._reduction = reduction.upper()
        self._exclude = exclude

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        loss = F.square(label - pred)
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
