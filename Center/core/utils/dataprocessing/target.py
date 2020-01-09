import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet import nd


# object size-adaptive standard deviation 구하기
# https://en.wikipedia.org/wiki/Gaussian_function
def gaussian_radius(height=512, width=512, min_overlap=0.7):
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)

    temp = max(0, b1 ** 2 - 4 * a1 * c1)
    sq1 = np.sqrt(temp)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height

    temp = max(0, b2 ** 2 - 4 * a2 * c2)
    sq2 = np.sqrt(temp)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height

    temp = max(0, b3 ** 2 - 4 * a3 * c3)
    sq3 = np.sqrt(temp)
    r3 = (b3 + sq3) / 2

    return max(0, int(min(r1, r2, r3)))


def gaussian_2d(shape=(10, 10), sigma=1):
    m, n = [s // 2 for s in shape]
    y, x = np.mgrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return h

def draw_gaussian(heatmap, center_x, center_y, radius, k=1):

    diameter = 2 * radius + 1  # 홀수
    gaussian = gaussian_2d(shape=(diameter, diameter), sigma=diameter / 6)

    # 경계선에서 어떻게 처리 할지
    height, width = heatmap.shape[0:2]
    left, right = min(center_x, radius), min(width - center_x, radius + 1)
    top, bottom = min(center_y, radius), min(height - center_y, radius + 1)

    masked_heatmap = heatmap[center_y - top : center_y + bottom, center_x - left : center_x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    '''
    https://rfriend.tistory.com/290
    Python Numpy의 배열 indexing, slicing에서 유의해야할 것이 있다. 
    배열을 indexing 해서 얻은 객체는 복사(copy)가 된 독립된 객체가 아니며, 
    단지 원래 배열의 view 일 뿐이라는 점이다.  
    따라서 view를 새로운 값으로 변경시키면 원래의 배열의 값도 변경이 된다.
    따라서 아래와 같은 경우 masked_heatmap은 view 일뿐.
    '''
    # inplace 연산
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

# https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
class TargetGenerator(gluon.Block):

    def __init__(self, num_classes=3):
        super(TargetGenerator, self).__init__()
        self._num_classes = num_classes

    def forward(self, gt_boxes, gt_ids, output_width, output_height, ctx):

        if isinstance(gt_boxes, mx.nd.NDArray):
            gt_boxes = gt_boxes.asnumpy()
        if isinstance(gt_ids, mx.nd.NDArray):
            gt_ids = gt_ids.asnumpy()

        batch_size = gt_boxes.shape[0]
        heatmap = np.zeros((batch_size, self._num_classes, output_height, output_width),
                           dtype=np.float32)
        offset_target = np.zeros((batch_size, 2, output_height, output_width), dtype=np.float32)
        wh_target = np.zeros((batch_size, 2, output_height, output_width), dtype=np.float32)
        mask_target = np.zeros((batch_size, 2, output_height, output_width), dtype=np.float32)

        for batch, gt_box, gt_id in zip(range(len(gt_boxes)), gt_boxes, gt_ids):
            for bbox, id in zip(gt_box, gt_id):

                # background인 경우
                if bbox[0] == -1 or bbox[1] == -1 or bbox[2] == -1 or bbox[3] == -1 or id == -1:
                    continue

                box_h, box_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                center = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    dtype=np.float32)
                center_int = center.astype(np.int32)
                center_x, center_y = center_int
                # data augmentation으로 인해 범위가 넘어갈수 가 있음.
                center_x = np.clip(center_x, 0, output_width-1)
                center_y = np.clip(center_y, 0, output_height-1)

                # heatmap
                # C:\ProgramData\Anaconda3\Lib\site-packages\gluoncv\model_zoo\center_net\target_generator.py
                radius = gaussian_radius(height=box_h, width=box_w)
                radius = max(0, int(radius))

                # 가우시안 그리기 - inplace 연산(np.maximum)
                draw_gaussian(heatmap[batch, int(id), ...], center_x, center_y,
                              radius)
                # wh
                box = np.array([box_w, box_h], dtype=np.float32)
                wh_target[batch, :, center_y, center_x] = box

                # offset
                offset_target[batch, :, center_y, center_x] = center - center_int

                # mask
                mask_target[batch, :, center_y, center_x] = 1.0

        return tuple([nd.array(ele, ctx=ctx) for ele in (heatmap, offset_target, wh_target, mask_target)])

# test
if __name__ == "__main__":
    from core import CenterTrainTransform, DetectionDataset
    import os

    input_size = (768, 1280)
    scale_factor = 4
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = CenterTrainTransform(input_size, mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'valid'), transform=transform)

    num_classes = dataset.num_class
    image, label, _ = dataset[0]
    targetgenerator = TargetGenerator(num_classes=num_classes)

    # batch 형태로 만들기
    label = np.expand_dims(label, axis=0)
    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    heatmap_target, offset_target, wh_target, mask_target = targetgenerator(gt_boxes, gt_ids, input_size[1] // scale_factor, input_size[0] // scale_factor, mx.cpu())

    print(f"heatmap_targets shape : {heatmap_target.shape}")
    print(f"offset_targets shape : {offset_target.shape}")
    print(f"wh_targets shape : {wh_target.shape}")
    print(f"mask_targets shape : {mask_target.shape}")
    '''
    heatmap_targets shape : (1, 5, 128, 128)
    offset_targets shape : (1, 2, 128, 128)
    wh_targets shape : (1, 2, 128, 128)
    mask_targets shape : (1, 2, 128, 128)
    '''
