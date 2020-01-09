import glob
import logging
import os
import platform

import cv2
import mxnet as mx
import mxnet.gluon as gluon
from tqdm import tqdm

from core import Prediction
from core import box_resize
from core import plot_bbox
from core import testdataloader

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)

def run(image_list=False,
        image_path="",
        weight_path="weights",
        load_name="512_512_ADAM_PRES_18",
        load_period=10, GPU_COUNT=1,
        decode_number=5000,
        multiperclass=True,
        nms_thresh=0.5,
        nms_topk=500,
        except_class_thresh=0.05,
        plot_class_thresh=0.5,
        image_save_path="result_image",
        image_show=True,
        image_save=True):
    if GPU_COUNT <= 0:
        ctx = mx.cpu(0)
    elif GPU_COUNT > 0:
        ctx = mx.gpu(0)

    # 운영체제 확인
    if platform.system() == "Linux":
        logging.info(f"{platform.system()} OS")
    elif platform.system() == "Windows":
        logging.info(f"{platform.system()} OS")
    else:
        logging.info(f"{platform.system()} OS")

    if GPU_COUNT > 0:
        free_memory, total_memory = mx.context.gpu_memory_info(0)
        free_memory = round(free_memory / (1024 * 1024 * 1024), 2)
        total_memory = round(total_memory / (1024 * 1024 * 1024), 2)
        logging.info(f'Running on {ctx} / free memory : {free_memory}GB / total memory {total_memory}GB')
    else:
        logging.info(f'Running on {ctx}')

    logging.info(f"test {load_name}")
    netheight = int(load_name.split("_")[0])
    netwidth = int(load_name.split("_")[1])
    if not isinstance(netheight, int) and not isinstance(netwidth, int):
        logging.info("height is not int")
        logging.info("width is not int")
        raise ValueError
    else:
        logging.info(f"network input size : {(netheight, netwidth)}")

    try:
        _, test_dataset = testdataloader()
    except Exception:
        logging.info("The dataset does not exist")
        exit(0)

    weight_path = os.path.join(weight_path, load_name)
    sym = os.path.join(weight_path, f'{load_name}_pre-symbol.json')
    params = os.path.join(weight_path, f'{load_name}_pre-{load_period:04d}.params')

    logging.info("symbol model test")
    try:
        net = gluon.SymbolBlock.imports(sym,
                                        ['data'],
                                        params, ctx=ctx)
    except Exception:
        # DEBUG, INFO, WARNING, ERROR, CRITICAL 의 5가지 등급
        logging.info("loading symbol weights 실패")
        exit(0)
    else:
        logging.info("loading symbol weights 성공")

    net.hybridize(active=True, static_alloc=True, static_shape=True)

    # BoxEncoder, BoxDecoder 에서 같은 값을 가져야함
    prediction = Prediction(
        from_sigmoid=False,
        num_classes=test_dataset.num_class,
        decode_number=decode_number,
        nms_thresh=nms_thresh,
        nms_topk=nms_topk,
        except_class_thresh=except_class_thresh,
        multiperclass=multiperclass)

    if image_list:
        types = ('*.jpg', '*.png')
        images_path = []
        for type in types:
            images_path.extend(glob.glob(os.path.join(image_path, type)))
        if images_path:
            for img_path in tqdm(images_path):
                name = os.path.splitext(os.path.basename(img_path))[0]
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                height, width, _ = image.shape
                logging.info(f"real input size : {(height, width)}")
                origin_image = image.copy()
                image = cv2.resize(image, (netwidth, netheight), interpolation=3)
                image[:, :, (0, 1, 2)] = image[:, :, (2, 1, 0)]  # BGR to RGB
                image = mx.nd.array(image, ctx=ctx)
                image = image.expand_dims(axis=0)

                cls_preds, box_preds, anchors = net(image)
                ids, scores, bboxes = prediction(cls_preds, box_preds, anchors)

                bbox = box_resize(bboxes[0], (netwidth, netheight), (width, height))
                plot_bbox(origin_image, bbox, scores=scores[0], labels=ids[0], thresh=plot_class_thresh,
                          reverse_rgb=False,
                          class_names=test_dataset.classes,
                          image_show=image_show,
                          image_save=image_save,
                          image_save_path=image_save_path,
                          image_name=name)
        else:
            raise FileNotFoundError
    else:
        name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        height, width, _ = image.shape
        logging.info(f"real input size : {(height, width)}")
        origin_image = image.copy()
        image = cv2.resize(image, (netwidth, netheight), interpolation=3)
        image[:, :, (0, 1, 2)] = image[:, :, (2, 1, 0)]  # BGR to RGB
        image = mx.nd.array(image, ctx=ctx)
        image = image.expand_dims(axis=0)

        cls_preds, box_preds, anchors = net(image)
        ids, scores, bboxes = prediction(cls_preds, box_preds, anchors)

        bbox = box_resize(bboxes[0], (netwidth, netheight), (width, height))
        plot_bbox(origin_image, bbox, scores=scores[0], labels=ids[0], thresh=plot_class_thresh,
                  reverse_rgb=False,
                  class_names=test_dataset.classes,
                  image_show=image_show,
                  image_save=image_save,
                  image_save_path=image_save_path,
                  image_name=name)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(image_list=True,  # True일 때, 폴더에 있는 이미지(jpg)들 전부다 평가 / False일 때, 한장 평가
        # image_path='Dataset/test/2. csm_meng_meng_baby_1_88cad0f74f.jpg',
        image_path='Dataset/test',
        weight_path="weights",
        load_name="512_512_ADAM_PRES_18",
        load_period=200, GPU_COUNT=0,
        decode_number=5000,
        multiperclass=True,
        nms_thresh=0.5,
        nms_topk=200,
        except_class_thresh=0.05,
        plot_class_thresh=0.5,
        image_save_path="result_image",
        image_show=False,
        image_save=True)
