# Gluoncv Loss에서 수정함.
from mxnet.gluon import HybridBlock


class HeatmapFocalLoss(HybridBlock):

    def __init__(self, from_sigmoid=True, alpha=2, beta=4):
        super(HeatmapFocalLoss, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._alpha = alpha
        self._beta = beta

    def hybrid_forward(self, F, pred, label):
        if not self._from_sigmoid:
            pred = F.sigmoid(pred)

        # a penalty-reduced pixelwise logistic regression with focal loss
        condition = label == 1
        loss = F.where(condition=condition,
                       x=F.power(1 - pred, self._alpha) * F.log(pred),
                       y=F.power(1 - label, self._beta) * F.power(pred, self._alpha) * F.log(1 - pred))

        norm = F.sum(condition).clip(1, 1e30)
        return -F.sum(loss) / norm


class NormedL1Loss(HybridBlock):

    def __init__(self):
        super(NormedL1Loss, self).__init__()

    def hybrid_forward(self, F, pred, label, mask):
        loss = F.abs(label * mask - pred * mask)
        norm = F.sum(mask).clip(1, 1e30)
        return F.sum(loss) / norm
