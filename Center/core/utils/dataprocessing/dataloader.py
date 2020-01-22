import mxnet as mx
import numpy as np
from gluoncv.data.dataloader import RandomTransformDataLoader
from mxnet.gluon.data import DataLoader

from core.utils.dataprocessing.dataset import DetectionDataset
from core.utils.dataprocessing.transformer import CenterTrainTransform, CenterValidTransform


class Tuple(object):

    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn,) + args
        for i, ele_fn in enumerate(self._fn):
            assert hasattr(ele_fn, '__call__'), 'Batchify functions must be callable! ' \
                                                'type(fn[%d]) = %s' % (i, str(type(ele_fn)))

    def __call__(self, data):

        assert len(data[0]) == len(self._fn), \
            'The number of attributes in each data sample should contains' \
            ' {} elements, given {}.'.format(len(self._fn), len(data[0]))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
        return ret


def _pad_arrs_to_max_length(arrs, pad_axis, pad_val, use_shared_mem=True):
    if not isinstance(arrs[0], (mx.nd.NDArray, np.ndarray)):
        arrs = [np.asarray(ele) for ele in arrs]
    original_length = [ele.shape[pad_axis] for ele in arrs]
    max_size = max(original_length)
    ret_shape = list(arrs[0].shape)
    ret_shape[pad_axis] = max_size
    ret_shape = (len(arrs),) + tuple(ret_shape)
    if use_shared_mem:
        ret = mx.nd.full(shape=ret_shape, val=pad_val, ctx=mx.Context('cpu_shared', 0),
                         dtype=arrs[0].dtype)
        original_length = mx.nd.array(original_length, ctx=mx.Context('cpu_shared', 0),
                                      dtype=np.int32)
    else:
        ret = mx.nd.full(shape=ret_shape, val=pad_val, dtype=arrs[0].dtype)
        original_length = mx.nd.array(original_length, dtype=np.int32)

    # arrs -> (batch, max object number, 5)
    for i, arr in enumerate(arrs):
        if arr.shape[pad_axis] == max_size:
            ret[i] = arr
        else:
            ret[i:i + 1, 0:arr.shape[pad_axis], :] = arr
    return ret, original_length


class Pad(object):

    def __init__(self, axis=0, pad_val=0, ret_length=False):
        self._axis = axis
        assert isinstance(axis, int), 'axis must be an integer! ' \
                                      'Received axis=%s, type=%s.' % (str(axis),
                                                                      str(type(axis)))
        self._pad_val = pad_val
        self._ret_length = ret_length

    def __call__(self, data):

        if isinstance(data[0], (mx.nd.NDArray, np.ndarray, list)):
            padded_arr, original_length = _pad_arrs_to_max_length(data, self._axis,
                                                                  self._pad_val, True)
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr
        else:
            raise NotImplementedError


class Stack(object):

    def __init__(self, use_shared_mem=False):
        self._use_shared_mem = use_shared_mem

    def __call__(self, batch):
        if isinstance(batch[0], mx.nd.NDArray):  # mxnet ndarray
            if self._use_shared_mem:
                out = mx.nd.empty((len(batch),) + batch[0].shape, dtype=batch[0].dtype,
                                  ctx=mx.Context('cpu_shared', 0))
                return mx.nd.stack(*batch, axis=0, out=out)
            else:
                return mx.nd.stack(*batch, axis=0)
        elif isinstance(batch[0], str):  # str
            return batch
        else:
            out = np.asarray(batch)
            if self._use_shared_mem:
                return mx.nd.array(out, ctx=mx.Context('cpu_shared', 0))
            else:
                return mx.nd.array(out)


def traindataloader(multiscale=False, factor_scale=[8, 5], augmentation=True, path="Dataset/train",
                    input_size=(512, 512), batch_size=8, batch_interval=10, num_workers=4, shuffle=True,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale_factor=4, make_target=True):
    if multiscale:

        h_seed = input_size[0] // factor_scale[0]
        w_seed = input_size[1] // factor_scale[0]

        init = factor_scale[0] - (factor_scale[1] // 2)
        end = factor_scale[0] + (factor_scale[1] // 2)
        end = end + 1

        dataset = DetectionDataset(path=path)
        train_transform = [CenterTrainTransform([x * h_seed, x * w_seed], mean=mean, std=std, scale_factor=scale_factor,
                                                augmentation=augmentation, make_target=make_target,
                                                num_classes=dataset.num_class) for x in
                           range(init, end)]

        dataloader = RandomTransformDataLoader(
            train_transform, dataset, batch_size=batch_size, interval=batch_interval, last_batch='rollover',
            shuffle=shuffle, batchify_fn=Tuple(Stack(use_shared_mem=True),
                                               Pad(pad_val=-1),
                                               Stack(use_shared_mem=True),
                                               Stack(use_shared_mem=True),
                                               Stack(use_shared_mem=True),
                                               Stack(use_shared_mem=True),
                                               Stack(),
                                               Stack(use_shared_mem=True),
                                               Pad(pad_val=-1)),
            num_workers=num_workers)
    else:
        transform = CenterTrainTransform(input_size, mean=mean, std=std, scale_factor=scale_factor,
                                         augmentation=augmentation, make_target=make_target,
                                         num_classes=DetectionDataset(path=path).num_class)
        dataset = DetectionDataset(path=path, transform=transform)
        ''')
        batchify_fn 왜 필요하지?
        -> 각 데이터들의 박스 개수가 다르기 때문
            batchify_fn -> 순서대로 stack images, and pad labels, stacked file name
        '''
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            batchify_fn=Tuple(Stack(use_shared_mem=True),
                              Pad(pad_val=-1),
                              Stack(use_shared_mem=True),
                              Stack(use_shared_mem=True),
                              Stack(use_shared_mem=True),
                              Stack(use_shared_mem=True),
                              Stack()),
            last_batch='rollover',  # or "keep", "discard"
            num_workers=num_workers)

    return dataloader, dataset


def validdataloader(path="Dataset/valid", input_size=(512, 512),
                    batch_size=1, num_workers=4, shuffle=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                    scale_factor=4, make_target=True):
    transform = CenterValidTransform(input_size, mean=mean, std=std, scale_factor=scale_factor, make_target=make_target,
                                     num_classes=DetectionDataset(path=path).num_class)
    dataset = DetectionDataset(path=path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        batchify_fn=Tuple(Stack(use_shared_mem=True),
                          Pad(pad_val=-1),
                          Stack(use_shared_mem=True),
                          Stack(use_shared_mem=True),
                          Stack(use_shared_mem=True),
                          Stack(use_shared_mem=True),
                          Stack()),
        last_batch='rollover',  # or "keep", "discard"
        num_workers=num_workers)

    return dataloader, dataset


def testdataloader(path="Dataset/test", input_size=(512, 512),
                   num_workers=4, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale_factor=4):
    transform = CenterValidTransform(input_size, mean=mean, std=std, scale_factor=scale_factor)
    dataset = DetectionDataset(path=path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        batchify_fn=Tuple(Stack(use_shared_mem=True),
                          Pad(pad_val=-1),
                          Stack(),
                          Stack(use_shared_mem=True),
                          Pad(pad_val=-1)),
        num_workers=num_workers)
    return dataloader, dataset


# test
if __name__ == "__main__":
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataloader, dataset = validdataloader(path=os.path.join(root, "Dataset/valid"), input_size=(512, 512),
                                          batch_size=1, num_workers=4, shuffle=True, mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

    # for문 돌리기 싫으므로, iterator로 만든
    dataloader_iter = iter(dataloader)
    data, label, _, _, _, _, name = next(dataloader_iter)

    # 첫번째 이미지만 가져옴
    image = data[0]
    label = label[0]
    name = name[0]

    print(f"image shape : {image.shape}")
    print(f"label shape : {label.shape}")
    print(f"name : {name}")
