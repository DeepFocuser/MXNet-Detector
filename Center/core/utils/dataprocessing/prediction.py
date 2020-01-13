import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import MaxPool2D


class Prediction(HybridBlock):

    def __init__(self, batch_size=1, topk=100, scale=4.0, amp=False):
        super(Prediction, self).__init__()
        self._batch_size = batch_size
        self._topk = topk
        self._scale = scale
        self._amp = amp
        self._heatmap_nms = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding=(1, 1))

    def hybrid_forward(self, F, heatmap, offset, wh):

        '''
        The peak keypoint extraction serves
        as a sufficient NMS alternative and can be implemented efficiently on device using a 3 × 3 max pooling operation.
        '''
        keep = self._heatmap_nms(heatmap) == heatmap
        heatmap = F.broadcast_mul(keep, heatmap)

        if self._amp:
            floatdtype = "float16"
            heatmap = F.cast(heatmap, dtype=floatdtype)
            offset = F.cast(offset, dtype=floatdtype)
            wh = F.cast(wh, dtype=floatdtype)
        else:
            floatdtype = "float32"

        _, channel, height, width = heatmap.shape_array().split(num_outputs=4, axis=0)  # int64임
        # 상위 self._topk개만 뽑아내기
        scores, indices = heatmap.reshape((0, -1)).topk(k=self._topk, axis=-1, ret_typ='both',
                                                        is_ascend=False)  # (batch, channel * height * width)
        scores = scores.expand_dims(-1)

        indices = F.cast(indices, dtype='int64')
        ids = F.broadcast_div(indices, (height * width))  # 정수/정수 는 정수 = // 연산
        ids = F.cast(ids, floatdtype) # c++에서 float으로 받아오기 때문에!!! 형 변환 필요
        ids = ids.expand_dims(-1)

        '''
        박스 복구
        To limit the computational burden, we use a single size prediction  WxHx2
        for all object categories. 
        offset, wh에 해당 
        '''
        offset = offset.transpose((0, 2, 3, 1)).reshape(
            (0, -1, 2))  # (batch, x, y, channel) -> (batch, height*width, 2)
        wh = wh.transpose((0, 2, 3, 1)).reshape(
            (0, -1, 2))  # (batch, width, height, channel) -> (batch, height*width, 2)
        topk_indices = F.broadcast_mod(indices, (height * width))  # 클래스별 index

        # 2차원 복구
        topk_ys = F.broadcast_div(topk_indices, width)  # y축 index
        topk_xs = F.broadcast_mod(topk_indices, width)  # x축 index

        # https://mxnet.apache.org/api/python/docs/api/ndarray/ndarray.html?highlight=gather_nd#mxnet.ndarray.gather_nd
        # offset 에서 offset_xs를 index로 보고 뽑기 - gather_nd를 알고 나니 상당히 유용한 듯.
        # x index가 0번에 있고, y index가 1번에 있으므로!!!
        batch_indices = F.cast(F.arange(self._batch_size).slice_like(
            offset, axes=(0)).expand_dims(-1).repeat(repeats=self._topk, axis=-1), 'int64')  # (batch, self._topk)
        offset_xs_indices = F.zeros_like(batch_indices, dtype='int64')
        offset_ys_indices = F.ones_like(batch_indices, dtype='int64')
        offset_xs = F.concat(batch_indices, topk_indices, offset_xs_indices, dim=0).reshape((3, -1))
        offset_ys = F.concat(batch_indices, topk_indices, offset_ys_indices, dim=0).reshape((3, -1))

        xs = F.gather_nd(offset, offset_xs).reshape(
            (-1, self._topk))  # (batch, height*width, 2) / (3, self_batch_size*self._topk)
        ys = F.gather_nd(offset, offset_ys).reshape(
            (-1, self._topk))  # (batch, height*width, 2) / (3, self_batch_size*self._topk)
        topk_xs = F.cast(topk_xs, floatdtype) + xs
        topk_ys = F.cast(topk_ys, floatdtype) + ys
        w = F.gather_nd(wh, offset_xs).reshape(
            (-1, self._topk))  # (batch, height*width, 2) / (3, self_batch_size*self._topk)
        h = F.gather_nd(wh, offset_ys).reshape(
            (-1, self._topk))  # (batch, height*width, 2) / (3, self_batch_size*self._topk)

        half_w = w / 2
        half_h = h / 2
        bboxes = [topk_xs - half_w, topk_ys - half_h, topk_xs + half_w, topk_ys + half_h]  # 각각 (batch, self._topk)
        bboxes = F.concat(*[bbox.expand_dims(-1) for bbox in bboxes],
                          dim=-1)  # (batch, self._topk, 1) ->  (batch, self._topk, 4)



        return ids, scores, bboxes * self._scale


# test
if __name__ == "__main__":
    import os
    from collections import OrderedDict
    from core.model.Center import CenterNet

    input_size = (512, 512)
    scale_factor = 4
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    '''
    heatmap의 bias가 -2.19 인 이유는??? retinanet의 식과 같은데... 흠..
    For the final conv layer of the classification subnet, we set the bias initialization to b = − log((1 − π)/π),
    where π specifies that at the start of training every anchor should be labeled as foreground with confidence of ∼π.
    We use π = .01 in all experiments, although results are robust to the exact value. As explained in §3.3, 
    this initialization prevents the large number of background anchors from generating a large, 
    destabilizing loss value in the first iteration of training
    '''
    net = CenterNet(base=18,
                    heads=OrderedDict([
                        ('heatmap', {'num_output': 3, 'bias': -2.19}),
                        ('offset', {'num_output': 2}),
                        ('wh', {'num_output': 2})
                    ]),
                    head_conv_channel=64,
                    pretrained=False,
                    root=os.path.join(root, 'models'),
                    use_dcnv2=False, ctx=mx.cpu())

    net.hybridize(active=True, static_alloc=True, static_shape=True)
    prediction = Prediction(batch_size=8, topk=100, scale=scale_factor)
    heatmap, offset, wh = net(
        mx.nd.random_uniform(low=0, high=1, shape=(2, 3, input_size[0], input_size[1]), ctx=mx.cpu()))
    ids, scores, bboxes = prediction(heatmap, offset, wh)

    print(f"< input size(height, width) : {input_size} >")
    print(f"topk class id shape : {ids.shape}")
    print(f"topk class scores shape : {scores.shape}")
    print(f"topk box predictions shape : {bboxes.shape}")
    '''
    < input size(height, width) : (512, 512) >
    topk class id shape : (2, 100, 1)
    topk class scores shape : (2, 100, 1)
    topk box predictions shape : (2, 100, 4)
    '''
