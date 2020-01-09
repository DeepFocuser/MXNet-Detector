# Gluoncv Loss에서 수정함.
from mxnet.gluon import HybridBlock


class FocalLoss(HybridBlock):

    def __init__(self, alpha=0.25, gamma=2, sparse_label=True,
                 from_sigmoid=False, batch_axis=0, num_class=None,
                 eps=1e-12, reduction="sum", exclude=False):
        super(FocalLoss, self).__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._sparse_label = sparse_label
        if sparse_label and (not isinstance(num_class, int) or (num_class < 1)):
            raise ValueError("Number of class > 0 must be provided if sparse label is used.")
        self._from_sigmoid = from_sigmoid
        self._batch_axis = batch_axis
        self._num_class = num_class + 1  # background 포함
        self._eps = eps
        self._reduction = reduction.upper()
        self._exclude = exclude

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        '''
        논문에서,
        Extending the focal loss to the multi-class case is straightforward and
        works well; for simplicity we focus on the binary loss in this work.
        '''
        if not self._from_sigmoid:
            pred = F.sigmoid(pred)
        if self._sparse_label:  # Dataset에서 처리해주지 않는다면, 보통 sparse label임
            one_hot = F.one_hot(label, self._num_class)
        else:
            one_hot = label > 0

        one_hot = F.slice_axis(data=one_hot, axis=-1, begin=1, end=None)
        pt = F.where(one_hot, pred, 1 - pred)
        t = F.ones_like(one_hot)

        # in practice a a may be set by inverse class frequency or treated
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        loss = -alpha * ((1 - pt) ** self._gamma) * F.log(F.minimum(pt + self._eps, 1))

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
