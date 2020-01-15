import glob
import os
from xml.etree.ElementTree import parse

# import json
import mxnet as mx
import numpy as np
from mxnet.gluon.data import Dataset

from core.utils.util.box_utils import box_resize


# multiscale training 용
class DetectionDataset_V0(Dataset):
    """
    Parameters
    ----------
    path : str(jpg)
        Path to input image directory.
    input_size : tuple or list -> (height(int), width(int))
    transform : object
    mean : 이미지 정규화 한 뒤 뺄 값, Default [0.485, 0.456, 0.406]
    std : 이미지 정규화 한 뒤 나눌 값 Default [0.229, 0.224, 0.225]
    """
    CLASSES = ['meerkat', 'otter', 'panda', 'raccoon', 'pomeranian']

    def __init__(self, path='Dataset/train'):
        super(DetectionDataset_V0, self).__init__()
        self._name = os.path.basename(path)
        self._image_path_List = glob.glob(os.path.join(path, "*.jpg"))
        self._items = []
        self.itemname = []
        self._make_item_list()

    def _make_item_list(self):

        if self._image_path_List:
            for image_path in self._image_path_List:
                xml_path = image_path.replace(".jpg", ".xml")
                self._items.append((image_path, xml_path))

                # 이름 저장
                base_image = os.path.basename(image_path)
                name = os.path.splitext(base_image)[0]
                self.itemname.append(name)
        else:
            raise FileNotFoundError

    def __getitem__(self, idx):

        image_path, label_path = self._items[idx]
        image = mx.image.imread(image_path, flag=1, to_rgb=True)
        label = self._parsing(label_path)  # dtype을 float 으로 해야 아래 단계에서 편하다
        return image, label, self.itemname[idx]

    def _parsing(self, path):
        xml_list = []
        try:
            tree = parse(path)
            root = tree.getroot()
            object = root.findall("object")
            for ob in object:
                if ob.find("bndbox") != None:
                    bndbox = ob.find("bndbox")
                    xmin, ymin, xmax, ymax = [int(pos.text) for i, pos in enumerate(bndbox.iter()) if i > 0]

                    # or
                    # xmin = int(bndbox.findtext("xmin"))
                    # ymin = int(bndbox.findtext("ymin"))
                    # xmax = int(bndbox.findtext("xmax"))
                    # ymax = int(bndbox.findtext("ymax"))

                    select = ob.findtext("name")
                    if select == "meerkat":
                        classes = 0
                    elif select == "otter":
                        classes = 1
                    elif select == "panda":
                        classes = 2
                    elif select == "raccoon":
                        classes = 3
                    elif select == "pomeranian":
                        classes = 4
                    xml_list.append((xmin, ymin, xmax, ymax, classes))
                else:
                    '''
                        image만 있고 labeling 없는 데이터에 대비 하기 위함 - ssd, retinanet loss에는 아무런 영향이 없음.
                        yolo 대비용임
                    '''
                    print(f"{path} : Image는 있는데, labeling 이 없어요")
                    xml_list.append((-1, -1, -1, -1, -1))

        except Exception:
            print("이상 파일 : " + path)
            xml_list.append((-1, -1, -1, -1, -1))
            return np.array(xml_list, dtype="float32")  # 반드시 numpy여야함.
        else:
            return np.array(xml_list, dtype="float32")  # 반드시 numpy여야함.

    @property
    def classes(self):
        return self.CLASSES

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.CLASSES)

    def __str__(self):
        return self._name + " " + "dataset"

    def __len__(self):
        return len(self._items)


# 일반 training 용
# https://github.com/5taku/custom_object_detection
class DetectionDataset(Dataset):
    """
    Parameters
    ----------
    path : str(jpg)
        Path to input image directory.
    input_size : tuple or list -> (height(int), width(int))
    transform : object
    mean : 이미지 정규화 한 뒤 뺄 값, Default [0.485, 0.456, 0.406]
    std : 이미지 정규화 한 뒤 나눌 값 Default [0.229, 0.224, 0.225]
    image_normalization : RetinaNet 학습시, True
    box_normalization : RetinaNet 학습시, True
    """
    CLASSES = ['meerkat', 'otter', 'panda', 'raccoon', 'pomeranian']

    def __init__(self, path='Dataset/train', input_size=(512, 512), mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform=None, image_normalization=True, box_normalization=True):
        super(DetectionDataset, self).__init__()
        self._name = os.path.basename(path)
        self._image_path_List = glob.glob(os.path.join(path, "*.jpg"))
        self._height = input_size[0]
        self._width = input_size[1]
        self._transform = transform
        self._items = []
        self._image_normalization = image_normalization
        self._box_normalization = box_normalization
        self.itemname = []
        self._mean = mean
        self._std = std
        self._make_item_list()

    def _make_item_list(self):

        if self._image_path_List:
            for image_path in self._image_path_List:
                xml_path = image_path.replace(".jpg", ".xml")
                self._items.append((image_path, xml_path))

                # 이름 저장
                base_image = os.path.basename(image_path)
                name = os.path.splitext(base_image)[0]
                self.itemname.append(name)
        else:
            raise FileNotFoundError

    def __getitem__(self, idx):

        image_path, label_path = self._items[idx]
        image = mx.image.imread(image_path, flag=1, to_rgb=True)
        label = self._parsing(label_path)  # dtype을 float 으로 해야 아래 단계에서 편하다

        h, w, _ = image.shape
        if self._transform is not None:
            result = self._transform(image, label)
            if len(result) == 2:
                image, label = result
                if self._image_normalization:
                    # c로 구현 한 것 파이썬으로 바인딩함 - 빠름
                    image = mx.nd.image.to_tensor(image)  # 0 ~ 1 로 바꾸기
                    image = mx.nd.image.normalize(image, mean=self._mean, std=self._std)
                if self._box_normalization:  # box normalization (range 0 ~ 1)
                    label[:, 0] = np.divide(label[:, 0], self._width)
                    label[:, 1] = np.divide(label[:, 1], self._height)
                    label[:, 2] = np.divide(label[:, 2], self._width)
                    label[:, 3] = np.divide(label[:, 3], self._height)
                    label = mx.nd.array(label)
                    return image, label, self.itemname[idx]
                else:
                    return image, label, self.itemname[idx]
            else:
                image, xcyc_target, wh_target, objectness, class_target, weights = result
                image = mx.nd.image.to_tensor(image)  # 0 ~ 1 로 바꾸기
                image = mx.nd.image.normalize(image, mean=self._mean, std=self._std)
                return image, xcyc_target, wh_target, objectness, class_target, weights, self.itemname[idx]
        else:
            image = mx.image.imresize(image, w=self._width, h=self._height, interp=3)
            label = box_resize(label, (w, h), (self._width, self._height))
            if self._image_normalization:
                image = mx.nd.image.to_tensor(image)  # 0 ~ 1 로 바꾸기
                image = mx.nd.image.normalize(image, mean=self._mean, std=self._std)
            if self._box_normalization:  # box normalization (range 0 ~ 1)
                label[:, 0] = np.divide(label[:, 0], self._width)
                label[:, 1] = np.divide(label[:, 1], self._height)
                label[:, 2] = np.divide(label[:, 2], self._width)
                label[:, 3] = np.divide(label[:, 3], self._height)
                label = mx.nd.array(label)
                return image, label, self.itemname[idx]
            else:
                label = mx.nd.array(label)
                return image, label, self.itemname[idx]

    def _parsing(self, path):
        xml_list = []
        try:
            tree = parse(path)
            root = tree.getroot()
            object = root.findall("object")
            for ob in object:
                if ob.find("bndbox") != None:
                    bndbox = ob.find("bndbox")
                    xmin, ymin, xmax, ymax = [int(pos.text) for i, pos in enumerate(bndbox.iter()) if i > 0]

                    # or
                    # xmin = int(bndbox.findtext("xmin"))
                    # ymin = int(bndbox.findtext("ymin"))
                    # xmax = int(bndbox.findtext("xmax"))
                    # ymax = int(bndbox.findtext("ymax"))

                    select = ob.findtext("name")
                    if select == "meerkat":
                        classes = 0
                    elif select == "otter":
                        classes = 1
                    elif select == "panda":
                        classes = 2
                    elif select == "raccoon":
                        classes = 3
                    elif select == "pomeranian":
                        classes = 4
                    xml_list.append((xmin, ymin, xmax, ymax, classes))
                else:
                    '''
                        image만 있고 labeling 없는 데이터에 대비 하기 위함 - ssd, retinanet loss에는 아무런 영향이 없음.
                        yolo 대비용임
                    '''
                    print(f"{path} : Image는 있는데, labeling 이 없어요")
                    xml_list.append((-1, -1, -1, -1, -1))

        except Exception:
            print("이상 파일 : " + path)
            xml_list.append((-1, -1, -1, -1, -1))
            return np.array(xml_list, dtype="float32")  # 반드시 numpy여야함.
        else:
            return np.array(xml_list, dtype="float32")  # 반드시 numpy여야함.

    @property
    def classes(self):
        return self.CLASSES

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.CLASSES)

    def __str__(self):
        return self._name + " " + "dataset"

    def __len__(self):
        return len(self._items)


# test용
class DetectionDataset_V1(Dataset):
    """
    Parameters
    ----------
    path : str(jpg)
        Path to input image directory.
    input_size : tuple or list -> (height(int), width(int))
    transform : object
    mean : 이미지 정규화 한 뒤 뺄 값, Default [0.485, 0.456, 0.406]
    std : 이미지 정규화 한 뒤 나눌 값 Default [0.229, 0.224, 0.225]
    image_normalization : RetinaNet 학습시, True
    box_normalization : RetinaNet 학습시, True
    """
    CLASSES = ['meerkat', 'otter', 'panda', 'raccoon', 'pomeranian']

    def __init__(self, path='Dataset/train', input_size=(512, 512), mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform=None, image_normalization=True, box_normalization=False):
        super(DetectionDataset_V1, self).__init__()
        self._name = os.path.basename(path)
        self._image_path_List = glob.glob(os.path.join(path, "*.jpg"))
        self._height = input_size[0]
        self._width = input_size[1]
        self._transform = transform
        self._items = []
        self._image_normalization = image_normalization
        self._box_normalization = box_normalization
        self.itemname = []
        self._mean = mean
        self._std = std
        self._make_item_list()

    def _make_item_list(self):

        if self._image_path_List:
            for image_path in self._image_path_List:
                xml_path = image_path.replace(".jpg", ".xml")
                self._items.append((image_path, xml_path))

                # 이름 저장
                base_image = os.path.basename(image_path)
                name = os.path.splitext(base_image)[0]
                self.itemname.append(name)
        else:
            raise FileNotFoundError

    def __getitem__(self, idx):

        image_path, label_path = self._items[idx]
        image = mx.image.imread(image_path, flag=1, to_rgb=True)
        origin_image = image.copy()
        label = self._parsing(label_path)  # dtype을 float 으로 해야 아래 단계에서 편하다
        origin_label = label.copy()

        h, w, _ = image.shape
        image = mx.image.imresize(image, w=self._width, h=self._height, interp=3)
        label = box_resize(label, (w, h), (self._width, self._height))
        if self._image_normalization:
            image = mx.nd.image.to_tensor(image)  # 0 ~ 1 로 바꾸기
            image = mx.nd.image.normalize(image, mean=self._mean, std=self._std)
        if self._box_normalization:  # box normalization (range 0 ~ 1)
            label[:, 0] = np.divide(label[:, 0], self._width)
            label[:, 1] = np.divide(label[:, 1], self._height)
            label[:, 2] = np.divide(label[:, 2], self._width)
            label[:, 3] = np.divide(label[:, 3], self._height)
            label = mx.nd.array(label)
            origin_label = mx.nd.array(origin_label)
            return image, label, origin_image, origin_label, self.itemname[idx]
        else:
            label = mx.nd.array(label)
            origin_label = mx.nd.array(origin_label)
            return image, label, origin_image, origin_label, self.itemname[idx]

    def _parsing(self, path):
        xml_list = []
        try:
            tree = parse(path)
            root = tree.getroot()
            object = root.findall("object")
            for ob in object:
                if ob.find("bndbox") != None:
                    bndbox = ob.find("bndbox")
                    xmin, ymin, xmax, ymax = [int(pos.text) for i, pos in enumerate(bndbox.iter()) if i > 0]

                    # or
                    # xmin = int(bndbox.findtext("xmin"))
                    # ymin = int(bndbox.findtext("ymin"))
                    # xmax = int(bndbox.findtext("xmax"))
                    # ymax = int(bndbox.findtext("ymax"))

                    select = ob.findtext("name")
                    if select == "meerkat":
                        classes = 0
                    elif select == "otter":
                        classes = 1
                    elif select == "panda":
                        classes = 2
                    elif select == "raccoon":
                        classes = 3
                    elif select == "pomeranian":
                        classes = 4
                    xml_list.append((xmin, ymin, xmax, ymax, classes))
                else:
                    '''
                        image만 있고 labeling 없는 데이터에 대비 하기 위함 - ssd, retinanet loss에는 아무런 영향이 없음.
                        yolo 대비용임
                    '''
                    print(f"{path} : Image는 있는데, labeling 이 없어요")
                    xml_list.append((-1, -1, -1, -1, -1))

        except Exception:
            print("이상 파일 : " + path)
            xml_list.append((-1, -1, -1, -1, -1))
            return np.array(xml_list, dtype="float32")  # 반드시 numpy여야함.
        else:
            return np.array(xml_list, dtype="float32")  # 반드시 numpy여야함.

    @property
    def classes(self):
        return self.CLASSES

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.CLASSES)

    def __str__(self):
        return self._name + " " + "dataset"

    def __len__(self):
        return len(self._items)


# test
if __name__ == "__main__":
    import random
    from core.utils.util.utils import plot_bbox

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), input_size=(512, 512),
                               transform=None, image_normalization=False, box_normalization=False)
    # dataset = DetectionDataset_V0(path=os.path.join(root, 'Dataset', 'train'))
    length = len(dataset)
    image, label, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('image shape:', image.shape)

    plot_bbox(image, label[:, :4],
              scores=None, labels=label[:, 4:5],
              class_names=dataset.classes, colors=None, reverse_rgb=True, absolute_coordinates=True,
              image_show=True, image_save=False, image_save_path="result", image_name=file_name)
