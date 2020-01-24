import os

import cv2

from core.utils.dataprocessing.target import TargetGenerator
from core.utils.util.box_utils import *
from core.utils.util.image_utils import *


class YoloTrainTransform(object):

    def __init__(self, height, width, net=None, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225], ignore_threshold=0.5, dynamic=True, from_sigmoid=False, augmentation=False,
                 make_target=False):

        self._height = height
        self._width = width
        self._mean = mean
        self._std = std
        self._augmentation = augmentation
        self._make_target = make_target
        if self._make_target:
            self._output1, self._output2, self._output3, self._anchor1, self._anchor2, self._anchor3, _, _, _, _, _, _ = net(
                mx.nd.zeros((1, 3, height, width)))
            self._target_generator = TargetGenerator(ignore_threshold=ignore_threshold,
                                                     dynamic=dynamic,
                                                     from_sigmoid=from_sigmoid)
        else:
            self._target_generator = None

    def __call__(self, img, bbox, name):

        if self._augmentation:
            # random color jittering - photo-metric distortions
            img = image_random_color_distort(img)

            # random expansion with prob 0.5
            expansion = np.random.choice([False, True], p=[0.5, 0.5])
            if expansion:
                # Random expand original image with borders, this is identical to placing the original image on a larger canvas.
                img, expand = random_expand(img, max_ratio=4, fill=[m * 255 for m in [0.485, 0.456, 0.406]],
                                            keep_ratio=True)
                bbox = box_translate(bbox, x_offset=expand[0], y_offset=expand[1], shape=img.shape[:-1])

            # random cropping
            h, w, _ = img.shape
            bbox, crop = box_random_crop_with_constraints(bbox, (w, h),
                                                          min_scale=0.1,
                                                          max_scale=1,
                                                          max_aspect_ratio=2,
                                                          constraints=None,
                                                          max_trial=50)

            x0, y0, w, h = crop
            img = mx.image.fixed_crop(img, x0, y0, w, h)

            # resize with random interpolation
            h, w, _ = img.shape
            interp = np.random.randint(0, 5)
            img = mx.image.imresize(img, self._width, self._height, interp=interp)
            bbox = box_resize(bbox, (w, h), (self._width, self._height))

            # random horizontal flip with probability of 0.5
            h, w, _ = img.shape
            img, flips = random_flip(img, px=0.5)
            bbox = box_flip(bbox, (w, h), flip_x=flips[0])

            # random vertical flip with probability of 0.5
            img, flips = random_flip(img, py=0.5)
            bbox = box_flip(bbox, (w, h), flip_y=flips[1])

            # random translation
            translation = np.random.choice([False, True], p=[0.5, 0.5])
            if translation:
                img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
                img = img.asnumpy()
                x_offset = np.random.randint(-20, high=20)
                y_offset = np.random.randint(-20, high=20)
                M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])  # +일 경우, (오른쪽, 아래)
                img = cv2.warpAffine(img, M, (w, h), borderValue=[m * 255 for m in [0.406, 0.456, 0.485]])
                bbox = box_translate(bbox, x_offset=x_offset, y_offset=y_offset, shape=(h, w))
                img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
                img = mx.nd.array(img)
        else:
            h, w, _ = img.shape
            img = mx.image.imresize(img, self._width, self._height, interp=1)
            bbox = box_resize(bbox, (w, h), (self._width, self._height))

        img = mx.nd.image.to_tensor(img)  # 0 ~ 1 로 바꾸기
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._make_target:
            bbox = bbox[np.newaxis, :, :]
            bbox = mx.nd.array(bbox)
            xcyc_target, wh_target, objectness, class_target, weights = self._target_generator(
                [self._output1, self._output2, self._output3],
                [self._anchor1, self._anchor2, self._anchor3],
                bbox[:, :, :4],
                bbox[:, :, 4:5],
                (self._height, self._width))
            return img, bbox[0], xcyc_target[0], wh_target[0], objectness[0], class_target[0], weights[0], name
        else:
            return img, bbox, name


class YoloValidTransform(object):

    def __init__(self, height, width, net=None, mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225], ignore_threshold=0.5, dynamic=True, from_sigmoid=False, make_target=False):
        self._height = height
        self._width = width
        self._mean = mean
        self._std = std
        self._make_target = make_target
        if self._make_target:
            self._output1, self._output2, self._output3, self._anchor1, self._anchor2, self._anchor3, _, _, _, _, _, _ = net(
                mx.nd.zeros((1, 3, height, width)))
            self._target_generator = TargetGenerator(ignore_threshold=ignore_threshold,
                                                     dynamic=dynamic,
                                                     from_sigmoid=from_sigmoid)
        else:
            self._target_generator = None

    def __call__(self, img, bbox, name):
        # resize with random interpolation
        h, w, _ = img.shape
        img = mx.image.imresize(img, self._width, self._height, interp=1)
        bbox = box_resize(bbox, (w, h), (self._width, self._height))

        img = mx.nd.image.to_tensor(img)  # 0 ~ 1 로 바꾸기
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._make_target:
            bbox = bbox[np.newaxis, :, :]
            bbox = mx.nd.array(bbox)
            xcyc_target, wh_target, objectness, class_target, weights = self._target_generator(
                [self._output1, self._output2, self._output3],
                [self._anchor1, self._anchor2, self._anchor3],
                bbox[:, :, :4],
                bbox[:, :, 4:5],
                (self._height, self._width))
            return img, bbox[0], xcyc_target[0], wh_target[0], objectness[0], class_target[0], weights[0], name
        else:
            return img, bbox, name


# test
if __name__ == "__main__":
    import random
    from core.utils.dataprocessing.dataset import DetectionDataset

    input_size = (320, 640)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = YoloTrainTransform(input_size[0], input_size[1], make_target=False)
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), transform=transform)

    length = len(dataset)
    image, label, _, _, _ = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('image shape:', image.shape)
    print('label shape:', label.shape)
