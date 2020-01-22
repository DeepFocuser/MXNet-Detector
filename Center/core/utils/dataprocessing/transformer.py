import os

from core.utils.dataprocessing.target import TargetGenerator
from core.utils.util.box_utils import *
from core.utils.util.image_utils import *


class CenterTrainTransform(object):

    def __init__(self, input_size, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), scale_factor=4, augmentation=True,
                 make_target=False, num_classes=3):

        self._width = input_size[1]
        self._height = input_size[0]
        self._mean = mean
        self._std = std
        self._scale_factor = scale_factor
        self._augmentation = augmentation
        self._make_target = make_target
        if self._make_target:
            self._target_generator = TargetGenerator(num_classes=num_classes)
        else:
            self._target_generator = None

    def __call__(self, img, bbox, name):

        output_w = self._width // self._scale_factor
        output_h = self._height // self._scale_factor

        if self._augmentation:

            # random color jittering
            distortion = np.random.choice([False, True], p=[0.7, 0.3])
            if distortion:
                img = image_random_color_distort(img)

            # random expansion with prob 0.5
            expansion = np.random.choice([False, True], p=[0.7, 0.3])
            if expansion:
                # Random expand original image with borders, this is identical to placing the original image on a larger canvas.
                img, expand = random_expand(img, max_ratio=4, fill=[m * 255 for m in [0., 0., 0.]],
                                            keep_ratio=True)
                bbox = box_translate(bbox, x_offset=expand[0], y_offset=expand[1], shape=img.shape[:-1])

            # random cropping
            random_crop = np.random.choice([False, True], p=[0.7, 0.3])
            if random_crop:
                h, w, _ = img.shape
                bbox, crop = box_random_crop_with_constraints(bbox, (w, h),
                                                              min_scale=0.3,
                                                              max_scale=1,
                                                              max_aspect_ratio=3,
                                                              constraints=None,
                                                              max_trial=50)

                x0, y0, w, h = crop
                img = mx.image.fixed_crop(img, x0, y0, w, h)

            # random horizontal flip with probability of 0.5
            h, w, _ = img.shape
            img, flips = random_flip(img, px=0.5)
            bbox = box_flip(bbox, (w, h), flip_x=flips[0])

            # random vertical flip with probability of 0.5
            img, flips = random_flip(img, py=0.5)
            bbox = box_flip(bbox, (w, h), flip_y=flips[1])

            # resize with random interpolation
            h, w, _ = img.shape
            interp = np.random.randint(0, 5)
            img = mx.image.imresize(img, self._width, self._height, interp=interp)
            bbox = box_resize(bbox, (w, h), (output_w, output_h))

        else:
            h, w, _ = img.shape
            img = mx.image.imresize(img, self._width, self._height, interp=1)
            bbox = box_resize(bbox, (w, h), (output_w, output_h))

        # heatmap 기반이기 때문에 제한 해줘야 한다.
        bbox[:, 0] = np.clip(bbox[:, 0], 0, output_w)
        bbox[:, 1] = np.clip(bbox[:, 1], 0, output_h)
        bbox[:, 2] = np.clip(bbox[:, 2], 0, output_w)
        bbox[:, 3] = np.clip(bbox[:, 3], 0, output_h)

        img = mx.nd.array(img)
        img = mx.nd.image.to_tensor(img)  # 0 ~ 1 로 바꾸기
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._make_target:
            bbox = bbox[np.newaxis, :, :]
            heatmap, offset_target, wh_target, mask_target = self._target_generator(bbox[:, :, :4], bbox[:, :, 4:5],
                                                                                    output_w, output_h, img.context)
            return img, bbox[0], heatmap[0], offset_target[0], wh_target[0], mask_target[0], name
        else:
            return img, bbox, name


class CenterValidTransform(object):

    def __init__(self, input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale_factor=4,
                 make_target=False, num_classes=3):
        self._width = input_size[1]
        self._height = input_size[0]
        self._mean = mean
        self._std = std
        self._scale_factor = scale_factor
        self._make_target = make_target
        if self._make_target:
            self._target_generator = TargetGenerator(num_classes=num_classes)
        else:
            self._target_generator = None

    def __call__(self, img, bbox, name):

        output_w = self._width // self._scale_factor
        output_h = self._height // self._scale_factor
        h, w, _ = img.shape
        img = mx.image.imresize(img, self._width, self._height, interp=1)  # Bilinear interpolation
        bbox = box_resize(bbox, (w, h), (output_w, output_h))

        img = mx.nd.array(img)
        img = mx.nd.image.to_tensor(img)  # 0 ~ 1 로 바꾸기
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._make_target:
            bbox = bbox[np.newaxis, :, :]
            heatmap, offset_target, wh_target, mask_target = self._target_generator(bbox[:, :, :4], bbox[:, :, 4:5],
                                                                                    output_w, output_h, img.context)

            return img, bbox[0], heatmap[0], offset_target[0], wh_target[0], mask_target[0], name
        else:
            return img, bbox, name


# test
if __name__ == "__main__":
    import random
    from core.utils.dataprocessing.dataset import DetectionDataset

    input_size = (960, 1280)
    scale_factor = 4
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    transform = CenterTrainTransform(input_size, mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225),
                                     scale_factor=scale_factor)
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), transform=transform)
    length = len(dataset)
    image, label, file_name, _, _ = dataset[random.randint(0, length - 1)]

    print('images length:', length)
    print('image shape:', image.shape)
    print('label shape:', label.shape)
