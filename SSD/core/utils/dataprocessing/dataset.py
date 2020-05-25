import glob
import logging
import os
from xml.etree.ElementTree import parse

import mxnet as mx
import numpy as np
from mxnet.gluon.data import Dataset

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
    CLASSES = ['meerkat', 'otter', 'panda', 'raccoon', 'pomeranian']

    def __init__(self, path='Dataset/train', transform=None):
        super(DetectionDataset, self).__init__()
        self._name = os.path.basename(path)
        self._image_path_List = glob.glob(os.path.join(path, "*.jpg"))
        self._transform = transform
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
                return result[0], result[1], result[2], result[3], result[4]
        else:
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
                    else:
                        xmin, ymin, xmax, ymax, classes = -1, -1, -1, -1, -1
                    xml_list.append((xmin, ymin, xmax, ymax, classes))
                else:
                    '''
                        image만 있고 labeling 없는 데이터에 대비 하기 위함 - ssd, retinanet loss에는 아무런 영향이 없음.
                        yolo 대비용임
                    '''
                    print(f"only image : {path}")
                    xml_list.append((-1, -1, -1, -1, -1))

        except Exception:
            print(f"only image or json crash : {path}")
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
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'))
    length = len(dataset)
    image, label, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('image shape:', image.shape)

    plot_bbox(image, label[:, :4],
              scores=None, labels=label[:, 4:5],
              class_names=dataset.classes, colors=None, reverse_rgb=True, absolute_coordinates=True,
              image_show=True, image_save=False, image_save_path="result", image_name=file_name)
