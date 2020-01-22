import mxnet as mx
from mxnet.gluon import HybridBlock


class Decoder(HybridBlock):

    def __init__(self, from_sigmoid=False, num_classes=5, thresh=0.01, multiperclass=True):
        super(Decoder, self).__init__()
        self._from_sigmoid = from_sigmoid
        self._num_classes = num_classes
        self._num_pred = 5 + num_classes
        self._thresh = thresh
        self._multiperclass = multiperclass

    def hybrid_forward(self, F, output, anchor, offset, stride):

        # 자르기
        out = F.reshape(output, shape=(0, -3, -1, self._num_pred))  # (b, 169, 3, 10)
        xy_pred = out.slice_axis(axis=-1, begin=0, end=2)  # (b, 169, 3, 2)
        wh_pred = out.slice_axis(axis=-1, begin=2, end=4)  # (b, 169, 3, 2)
        objectness = out.slice_axis(axis=-1, begin=4, end=5)  # (b, 169, 3, 1)
        class_pred = out.slice_axis(axis=-1, begin=5, end=None)  # (b, 169, 3, 5)

        if not self._from_sigmoid:
            xy_pred = F.sigmoid(xy_pred)
            objectness = F.sigmoid(objectness)
            class_pred = F.sigmoid(class_pred)

        # 복구하기
        '''
        offset이 output에 따라 변하는 값이기 때문에, 
        네트워크에서 출력할 때 충분히 크게 만들면,
        c++에서 inference 할 때 어떤 값을 넣어도 정상적으로 동작하게 된다. 
        '''
        offset = F.slice_like(offset, shape_like=output, axes=(1, 2))
        offset = F.reshape(offset, shape=(0, -3, 0, 0))
        xy_preds = F.broadcast_mul(F.broadcast_add(xy_pred, offset), stride)
        wh_preds = F.broadcast_mul(F.exp(wh_pred), anchor)
        class_pred = F.broadcast_mul(class_pred, objectness)  # (b, 169, 3, 5)

        # center to corner
        wh = wh_preds / 2.0
        bbox = F.concat(xy_preds - wh, xy_preds + wh, dim=-1)  # (b, 169, 3, 4)

        # prediction per class
        if self._multiperclass:

            bbox = F.tile(bbox, reps=(self._num_classes, 1, 1, 1, 1))  # (5, b, 169, 3, 4)
            class_pred = F.transpose(class_pred, axes=(3, 0, 1, 2)).expand_dims(axis=-1)  # (5, b, 169, 3, 1)

            '''
            여기에서 들었던 의문, 
            block 안에서 데이터를 생성하면? ndarray의 경우 ctx를 지정 해줘야 하지 않나? with x.context as ctx 로 인해 생각할 필요 없다.
            
            상세 분석 : 아래와 같이 HybridBlock의 forward함수의 내부를 보면 x.context as ctx 로 default ctx를 
            x의 ctx로 지정해주기 때문에(Decoder클래스에서 x=output) 상관없다. 
            
            느낀점 : 조금은 부족한 점이 있지만, gluon block, HybridBlock을 꽤나 견고하게 만든듯 
            def forward(self, x, *args):
                """Defines the forward computation. Arguments can be either
                :py:class:`NDArray` or :py:class:`Symbol`."""
                if isinstance(x, NDArray):
                    with x.context as ctx:
                        if self._active:
                            return self._call_cached_op(x, *args)
        
                        try:
                            params = {i: j.data(ctx) for i, j in self._reg_params.items()}
                        except DeferredInitializationError:
                            self._deferred_infer_shape(x, *args)
                            for _, i in self.params.items():
                                i._finish_deferred_init()
                            params = {i: j.data(ctx) for i, j in self._reg_params.items()}
        
                        return self.hybrid_forward(ndarray, x, *args, **params)

            '''
            id = F.broadcast_add(class_pred * 0,
                                 F.arange(0, self._num_classes).reshape(
                                     (0, 1, 1, 1, 1)))  # (5, b, 169, 3, 1)

            # ex) thresh=0.01 이상인것만 뽑기
            mask = class_pred > self._thresh
            id = F.where(mask, id, F.ones_like(id) * -1)
            score = F.where(mask, class_pred, F.zeros_like(class_pred))

            # reshape to (b, -1, 6)
            results = F.concat(id, score, bbox, dim=-1)  # (5, b, 169, 3, 6)
            results = F.transpose(results, axes=(1, 0, 2, 3, 4))  # (5, b, 169, 3, 6) -> (b, 5, 169, 3, 6)
        else:  # prediction multiclass
            id = F.argmax(class_pred, axis=-1, keepdims=True)
            class_pred = F.pick(class_pred, id, axis=-1, keepdims=True)

            # ex) thresh=0.01 이상인것만 뽑기
            mask = class_pred > self._thresh
            id = F.where(mask, id, F.ones_like(id) * -1)
            score = F.where(mask, class_pred, F.zeros_like(class_pred))

            results = F.concat(id, score, bbox, dim=-1)  # (b, 169, 3, 6)
        return F.reshape(results, shape=(0, -1, 6))  # (b, -1, 6)


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

    # net.hybridize(active=True, static_alloc=True, static_shape=True)

    # batch 형태로 만들기
    image = image.expand_dims(axis=0)
    label = label.expand_dims(axis=0)

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(
        image)

    results = []
    decoder = Decoder(from_sigmoid=False, num_classes=num_classes, thresh=0.01, multiperclass=True)
    for out, an, off, st in zip([output1, output2, output3], [anchor1, anchor2, anchor3], [offset1, offset2, offset3],
                                [stride1, stride2, stride3]):
        results.append(decoder(out, an, off, st))
    results = mx.nd.concat(*results, dim=1)
    print(f"decoder shape : {results.shape}")
    '''
    multiperclass=True 일 때 
    decoder shape : (1, 53235, 6)
    
    multiperclass=False 일 때 
    decoder shape : (1, 10647, 6)
    '''
