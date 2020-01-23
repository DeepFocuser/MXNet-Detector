import mxnet as mx
import mxnet.autograd as autograd
import numpy as np
from mxnet.gluon import Block

from core.utils.dataprocessing.targetFunction.matching import Matcher


class BBoxSplit(Block):

    def __init__(self, axis, squeeze_axis=False):
        super(BBoxSplit, self).__init__()
        self._axis = axis
        self._squeeze_axis = squeeze_axis

    def forward(self, x):
        F = mx.nd
        return F.split(x, axis=self._axis, num_outputs=4, squeeze_axis=self._squeeze_axis)


class BBoxBatchIOU(Block):

    def __init__(self, axis=-1):
        super(BBoxBatchIOU, self).__init__()
        self._pre = BBoxSplit(axis=axis, squeeze_axis=True)

    def forward(self, a, b):
        F = mx.nd
        """Compute IOU for each batch
        Parameters
        ----------
        a : mxnet.nd.NDArray or mxnet.sym.Symbol
            (B, N, 4) first input.
        b : mxnet.nd.NDArray or mxnet.sym.Symbol
            (B, M, 4) second input.
        Returns
        -------
        mxnet.nd.NDArray or mxnet.sym.Symbol
            (B, N, M) array of IOUs.
        """
        al, at, ar, ab = self._pre(a)
        bl, bt, br, bb = self._pre(b)

        # (B, N, M)
        left = F.broadcast_maximum(al.expand_dims(-1), bl.expand_dims(-2))
        right = F.broadcast_minimum(ar.expand_dims(-1), br.expand_dims(-2))
        top = F.broadcast_maximum(at.expand_dims(-1), bt.expand_dims(-2))
        bot = F.broadcast_minimum(ab.expand_dims(-1), bb.expand_dims(-2))

        # clip with (0, float16.max)
        iw = F.clip(right - left, a_min=0, a_max=6.55040e+04)
        ih = F.clip(bot - top, a_min=0, a_max=6.55040e+04)
        i = iw * ih

        # areas
        area_a = ((ar - al) * (ab - at)).expand_dims(-1)
        area_b = ((br - bl) * (bb - bt)).expand_dims(-2)
        union = F.broadcast_add(area_a, area_b) - i
        return i / union


class BBoxCornerToCenter(Block):
    def __init__(self, axis=-1):
        super(BBoxCornerToCenter, self).__init__()
        self._axis = axis

    def forward(self, x):
        F = mx.nd
        xmin, ymin, xmax, ymax = F.split(x, axis=self._axis, num_outputs=4)
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + width / 2
        y_center = ymin + height / 2
        return x_center, y_center, width, height


class Encoderdynamic(Block):

    def __init__(self, ignore_threshold=0.7, from_sigmoid=False):
        super(Encoderdynamic, self).__init__()
        self._cornertocenter = BBoxCornerToCenter(axis=-1)
        self._batch_iou = BBoxBatchIOU(axis=-1)
        self._from_sigmoid = from_sigmoid
        self._ignore_threshold = ignore_threshold

    def forward(self, matches, ious, outputs, anchors, gt_boxes, gt_ids, input_size):

        F = mx.nd
        in_height = input_size[0]
        in_width = input_size[1]
        feature_size = []
        anchor_size = []
        strides = []

        for i, out, anchor in zip(range(len(outputs)), outputs, anchors):
            _, h, w, ac = out.shape
            _, _, a, _ = anchor.shape
            feature_size.append([h, w])
            anchor_size.append(a)
            stride = np.reshape([in_width // w, in_height // h], (1, 1, 1, 2))
            strides.append(stride)

        self._num_pred = np.divide(ac, a).astype(int)
        all_anchors = F.concat(*[anchor.reshape(-1, 2) for anchor in anchors], dim=0)
        num_anchors = np.cumsum([anchor_size for anchor_size in anchor_size])  # ex) (3, 6, 9)
        num_offsets = np.cumsum([np.prod(feature) for feature in feature_size])  # ex) (338, 1690, 3549)
        offsets = [0] + num_offsets.tolist()

        # target 공간 만들어 놓기
        xcyc_targets = F.zeros(shape=(gt_boxes.shape[0],
                                      num_offsets[-1],
                                      num_anchors[-1], 2),
                               ctx=gt_boxes.context,
                               dtype=gt_boxes.dtype)  # (batch, 3549, 9, 2)가 기본 요소
        wh_targets = F.zeros_like(xcyc_targets)
        weights = F.zeros_like(xcyc_targets)
        # 1 : object, 0 : no object, -1 : ignore
        objectness = F.zeros_like(xcyc_targets.split(axis=-1, num_outputs=2)[0])
        class_targets = F.zeros_like(objectness)

        gtx, gty, gtw, gth = self._cornertocenter(gt_boxes)
        np_gtx, np_gty, np_gtw, np_gth = [x.asnumpy() for x in [gtx, gty, gtw, gth]]
        np_anchors = all_anchors.asnumpy()
        np_gt_ids = gt_ids.asnumpy()
        np_gt_ids = np_gt_ids.astype(int)

        # 가장 큰것에 할당하고, target network prediction 비교해서 0.7 이상인것들 무시하기
        batch, anchorN, objectN = ious.shape
        for b in range(batch):
            for a in range(anchorN):
                for o in range(objectN):
                    nlayer = np.where(num_anchors > a)[0][0]
                    out_height = outputs[nlayer].shape[1]
                    out_width = outputs[nlayer].shape[2]

                    gtx, gty, gtw, gth = (np_gtx[b, o, 0], np_gty[b, o, 0],
                                          np_gtw[b, o, 0], np_gth[b, o, 0])
                    ''' 
                        matching 단계에서 image만 들어온 데이터들도 matching이 되기때문에 아래와 같이 걸러줘야 한다.
                        image만 들어온 데이터 or padding 된것들은 noobject이다. 
                    '''
                    if gtx == -1.0 and gty == -1.0 and gtw == 0.0 and gth == 0.0:
                        continue
                    # compute the location of the gt centers
                    loc_x = int(gtx / in_width * out_width)
                    loc_y = int(gty / in_height * out_height)
                    # write back to targets
                    index = offsets[nlayer] + loc_y * out_width + loc_x
                    if a == matches[b, o]:  # 최대인 값은 제외
                        xcyc_targets[b, index, a, 0] = gtx / in_width * out_width - loc_x
                        xcyc_targets[b, index, a, 1] = gty / in_height * out_height - loc_y
                        '''
                        if gtx == -1.0 and gty == -1.0 and gtw == 0.0 and gth == 0.0: 
                            continue
                        에서 처리를 해주었으나, 그래도 한번 더 대비 해놓자.
                        '''
                        wh_targets[b, index, a, 0] = np.log(
                            max(gtw, 1) / np_anchors[a, 0])  # max(gtw,1)? gtw,gth가 0일경우가 있다.
                        wh_targets[b, index, a, 1] = np.log(max(gth, 1) / np_anchors[a, 1])
                        weights[b, index, a, :] = 2.0 - gtw * gth / in_width / in_height
                        objectness[b, index, a, 0] = 1
                        class_targets[b, index, a, 0] = np_gt_ids[b, o, 0]

        xcyc_targets = self._slice(xcyc_targets, num_anchors, num_offsets)
        wh_targets = self._slice(wh_targets, num_anchors, num_offsets)
        weights = self._slice(weights, num_anchors, num_offsets)
        objectness = self._slice(objectness, num_anchors, num_offsets)
        class_targets = self._slice(class_targets, num_anchors, num_offsets)
        class_targets = F.squeeze(class_targets, axis=-1)

        '''
        언제 with autograd.pause():를 씌워 주는가?
        self._dynamic=True 일 때, 처음에 학습이 안되서 error log를 분석하다가 backward에 집중하게 됬다.

        segmentation fault: 11

        Stack trace:
          [bt] (0) /home/jg/anaconda3/envs/mxnetcuda/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2e6b420) [0x7f7b81092420]
          [bt] (1) /lib/x86_64-linux-gnu/libc.so.6(+0x354b0) [0x7f7bdad2f4b0]
          [bt] (2) /home/jg/anaconda3/envs/mxnetcuda/lib/python3.6/site-packages/mxnet/libmxnet.so(mxnet::imperative::SetDependency(nnvm::NodeAttrs const&, mxnet::Context const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, std::vector<mxnet::engine::Var*, std::allocator<mxnet::engine::Var*> >*, std::vector<mxnet::engine::Var*, std::allocator<mxnet::engine::Var*> >*, std::vector<mxnet::Resource, std::allocator<mxnet::Resource> >*, std::vector<unsigned int, std::allocator<unsigned int> >*, mxnet::DispatchMode)+0x271) [0x7f7b80891cc1]
          [bt] (3) /home/jg/anaconda3/envs/mxnetcuda/lib/python3.6/site-packages/mxnet/libmxnet.so(mxnet::Imperative::InvokeOp(mxnet::Context const&, nnvm::NodeAttrs const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, std::vector<mxnet::OpReqType, std::allocator<mxnet::OpReqType> > const&, mxnet::DispatchMode, mxnet::OpStatePtr)+0x18a) [0x7f7b8089318a]
          [bt] (4) /home/jg/anaconda3/envs/mxnetcuda/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2676d20) [0x7f7b8089dd20]
          [bt] (5) /home/jg/anaconda3/envs/mxnetcuda/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2677b48) [0x7f7b8089eb48]
          [bt] (6) /home/jg/anaconda3/envs/mxnetcuda/lib/python3.6/site-packages/mxnet/libmxnet.so(mxnet::imperative::RunGraph(bool, nnvm::IndexedGraph const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, unsigned long, unsigned long, std::vector<mxnet::OpReqType, std::allocator<mxnet::OpReqType> >&&, std::vector<unsigned int, std::allocator<unsigned int> >&&, std::vector<mxnet::OpStatePtr, std::allocator<mxnet::OpStatePtr> >*, std::vector<mxnet::DispatchMode, std::allocator<mxnet::DispatchMode> > const&, bool, std::vector<mxnet::TShape, std::allocator<mxnet::TShape> >*)+0x1f2) [0x7f7b8089fdd2]
          [bt] (7) /home/jg/anaconda3/envs/mxnetcuda/lib/python3.6/site-packages/mxnet/libmxnet.so(mxnet::Imperative::Backward(std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, std::vector<mxnet::NDArray*, std::allocator<mxnet::NDArray*> > const&, bool, bool, bool)+0x3b75) [0x7f7b8089af25]
          [bt] (8) /home/jg/anaconda3/envs/mxnetcuda/lib/python3.6/site-packages/mxnet/libmxnet.so(MXAutogradBackwardEx+0x573) [0x7f7b8078a043]

        self._encode를 보면 network에서 나온 출력이 들어간다, 이말은 즉슨, self._encoder(Encoderdynamic)도 자동으로 미분해버린다는 얘기 인데, 이게 미분이 가능한 구조가 아니다.
        따라서 autograd.pause()를 사용해서 gradient 계산을 정지 시켜야 한다.
        '''
        with autograd.pause():
            # dynamic
            box_preds = []
            for out, an, st in zip(outputs, anchors, strides):
                box_preds.append(self._boxdecoder(out, an, st))
            box_preds = F.concat(*box_preds, dim=1)
            batch_ious = self._batch_iou(box_preds, gt_boxes)  # (b, N, M)
            ious_max = batch_ious.max(axis=-1, keepdims=True)  # (b, N, 1)
            objectness_dynamic = (ious_max > self._ignore_threshold) * -1  # ignore
            # objectness 와 objectness_dynamic 조합하기
            objectness = F.where(objectness > 0, objectness, objectness_dynamic)
            # threshold 바꿔가며 개수 세어보기
            return xcyc_targets, wh_targets, objectness, class_targets, weights

    def _slice(self, x, num_anchors, num_offsets):
        F = mx.nd
        anchors = [0] + num_anchors.tolist()
        offsets = [0] + num_offsets.tolist()
        ret = []
        for i in range(len(num_anchors)):
            y = x[:, offsets[i]:offsets[i + 1], anchors[i]:anchors[i + 1], :]
            ret.append(y.reshape(0, -3, -1))
        return F.concat(*ret, dim=1)

    def _boxdecoder(self, output, anchor, stride):

        F = mx.nd
        _, height, width, _ = output.shape
        stride = F.array(stride, ctx=output.context)
        # grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        grid_y, grid_x = np.mgrid[:height, :width]
        offset = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)  # (13,13,2)
        offset = np.reshape(offset, (1, -1, 1, 2))  # (1, 169, 1, 2)
        offset = F.array(offset, dtype=output.dtype, ctx=output.context)
        # 자르기
        output = F.reshape(output, shape=(0, -3, -1, self._num_pred))  # (b, 169, 3, 10)
        xy_pred = output.slice_axis(axis=-1, begin=0, end=2)  # (b, 169, 3, 2)
        wh_pred = output.slice_axis(axis=-1, begin=2, end=4)  # (b, 169, 3, 2)
        if not self._from_sigmoid:
            xy_pred = F.sigmoid(xy_pred)
        xy_preds = F.multiply(F.add(xy_pred, offset), stride)
        wh_preds = F.multiply(F.exp(wh_pred), anchor)
        # center to corner
        wh = F.divide(wh_preds, 2.0)
        bbox = F.concat(xy_preds - wh, xy_preds + wh, dim=-1)  # (b, 169, 3, 4)
        return bbox.reshape(0, -1, 4)


# test
if __name__ == "__main__":
    from core import Yolov3, YoloTrainTransform, DetectionDataset
    import os

    input_size = (416, 416)
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
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

    matcher = Matcher()
    encoder = Encoderdynamic(ignore_threshold=0.7, from_sigmoid=False)

    # batch 형태로 만들기
    image = image.expand_dims(axis=0)
    label = label.expand_dims(axis=0)

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    output1, output2, output3, anchor1, anchor2, anchor3, _, _, _, _, _, _ = net(
        image)

    matches, ious = matcher([anchor1, anchor2, anchor3], gt_boxes)
    xcyc_targets, wh_targets, objectness, class_targets, weights = encoder(matches, ious, [output1, output2, output3],
                                                                           [anchor1, anchor2, anchor3], gt_boxes,
                                                                           gt_ids,
                                                                           input_size)

    print(f"< input size(height, width) : {input_size} >")
    print(f"xcyc_targets shape : {xcyc_targets.shape}")
    print(f"wh_targets shape : {wh_targets.shape}")
    print(f"objectness shape : {objectness.shape}")
    print(f"class_targets shape : {class_targets.shape}")
    print(f"weights shape : {weights.shape}")
    '''
    < input size(height, width) : (416, 416) >
    xcyc_targets shape : (1, 10647, 2)
    wh_targets shape : (1, 10647, 2)
    objectness shape : (1, 10647, 1)
    class_targets shape : (1, 10647)
    weights shape : (1, 10647, 2)
    '''
