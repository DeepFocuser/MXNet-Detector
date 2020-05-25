import glob
import json
import os
import random

import mxnet as mx
import numpy as np
from mxnet.gluon.data import Dataset
import logging
from core.utils.util.utils import plot_bbox

'''
데이터셋 출처
https://bdd-data.berkeley.edu/
'''

box_size_limit = 32

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

class DetectionDataset(Dataset):
    """
    Parameters
    ----------
    path : str(jpg)
        Path to input image directory.
    transform : object
    """
    CLASSES = ["bus", "traffic light", "traffic sign", "person", "bike", "truck", "motor", "car", "train", "rider"]

    def __init__(self, path='Dataset/train', transform=None):
        super(DetectionDataset, self).__init__()
        self._name = os.path.basename(path)
        self._label_path_List = glob.glob(os.path.join(path, "*.json"))
        self._transform = transform
        self._items = []
        self.itemname = []
        self._make_item_list()

    def _make_item_list(self):

        if self._label_path_List:
            for label_path in self._label_path_List:
                image_path = label_path.replace(".json", ".jpg")
                self._items.append((image_path, label_path))

                # 이름 저장
                base_image = os.path.basename(image_path)
                name = os.path.splitext(base_image)[0]
                self.itemname.append(name)
        else:
            logging.info("The dataset does not exist")

    def __getitem__(self, idx):

        image_path, label_path = self._items[idx]
        image = mx.image.imread(image_path, flag=1, to_rgb=True)
        origin_image = image.copy()
        label = self._parsing(label_path)  # dtype을 float 으로 해야 아래 단계에서 편하다
        origin_label = label.copy()

        if self._transform:
            result = self._transform(image, label, self.itemname[idx])
            if len(result) == 3:
                return result[0], result[1], result[2], origin_image, origin_label
            else:
                return result[0], result[1], result[2], result[3], result[4], result[5], result[
                    6]
        else:
            return image, label, self.itemname[idx]

    def _parsing(self, path):
        json_list = []
        # json파일 parsing - 순서 -> topleft_x, topleft_y, bottomright_x, bottomright_y, center_x, center_y
        try:
            with open(path, mode='r') as json_file:
                dict = json.load(json_file)
            for label in dict["labels"]:
                if "box2d" in label.keys():
                    xmin = label["box2d"]["x1"]
                    ymin = label["box2d"]["y1"]
                    xmax = label["box2d"]["x2"]
                    ymax = label["box2d"]["y2"]
                    category = label["category"]

                    box_size = (ymax - ymin, xmax - xmin)
                    # 사이즈가 너무 작은 박스는 제외하자
                    if box_size[0] > box_size_limit and box_size[1] > box_size_limit:
                        if category == self.CLASSES[0]:  # bus
                            classes = 0
                        elif category == self.CLASSES[1]:  # traffic light
                            classes = 1
                        elif category == self.CLASSES[2]:  # traffic sign
                            classes = 2
                        elif category == self.CLASSES[3]:  # person
                            classes = 3
                        elif category == self.CLASSES[4]:  # bike
                            classes = 4
                        elif category == self.CLASSES[5]:  # truck
                            classes = 5
                        elif category == self.CLASSES[6]:  # motor
                            classes = 6
                        elif category == self.CLASSES[7]:  # car
                            classes = 7
                        elif category == self.CLASSES[8]:  # train
                            classes = 8
                        elif category == self.CLASSES[9]:  # rider
                            classes = 9
                        else:
                            xmin, ymin, xmax, ymax, classes = -1, -1, -1, -1, -1
                    else:
                        xmin, ymin, xmax, ymax, classes = -1,-1,-1,-1,-1
                    json_list.append((xmin, ymin, xmax, ymax, classes))
                else:
                    print(f"only image : {path}")
                    json_list.append((-1, -1, -1, -1, -1))
        except Exception:
            print(f"only image or json crash : {path}")
            json_list.append((-1, -1, -1, -1, -1))
            return np.array(json_list, dtype="float32")  # 반드시 numpy여야함.
        else:
            return np.array(json_list, dtype="float32")  # 반드시 numpy여야함.

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
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'))

    length = len(dataset)
    image, label, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('image shape:', image.shape)

    plot_bbox(image, label[:, :4],
              scores=None, labels=label[:, 4:5],
              class_names=dataset.classes, colors=None, reverse_rgb=True, absolute_coordinates=True,
              image_show=True, image_save=False, image_save_path="result", image_name=file_name)
