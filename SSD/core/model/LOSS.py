# Gluoncv Loss에서 수정함.

from mxnet.gluon import HybridBlock


class SoftmaxCrossEntropyLoss(HybridBlock):

    def __init__(self, axis=-1, sparse_label=True, from_log_softmax=False,
                 batch_axis=0, reduction="sum", exclude=False):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_log_softmax = from_log_softmax
        self._batch_axis = batch_axis
        self._reduction = reduction.upper()
        self._exclude = exclude

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_log_softmax:
            pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            loss = -F.sum(pred * label, axis=self._axis, keepdims=True)

        if sample_weight is not None:
            loss = F.broadcast_mul(loss, sample_weight)

        if self._reduction == "SUM":
            return F.sum(loss, axis=self._batch_axis, exclude=self._exclude)
        elif self._reduction == "MEAN":
            return F.mean(loss, axis=self._batch_axis, exclude=self._exclude)
        else:
            raise NotImplementedError


class HuberLoss(HybridBlock):

    def __init__(self, rho=1, batch_axis=0, reduction="sum", exclude=False):
        super(HuberLoss, self).__init__()
        self._rho = rho
        self._batch_axis = batch_axis
        self._reduction = reduction.upper()
        self._exclude = exclude

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        loss = F.abs(label - pred)
        loss = F.where(loss > self._rho, loss - 0.5 * self._rho,
                       (0.5 / self._rho) * F.square(loss))
        if sample_weight is not None:
            loss = F.broadcast_mul(loss, sample_weight)

        if self._reduction == "SUM":
            return F.sum(loss, axis=self._batch_axis, exclude=self._exclude)
        elif self._reduction == "MEAN":
            return F.mean(loss, axis=self._batch_axis, exclude=self._exclude)
        else:
            raise NotImplementedError
