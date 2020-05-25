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
from core import plot_bbox, export_block_for_cplusplus
from core import testdataloader

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def run(image_list=True,  # True일 때, 폴더에 있는 이미지(jpg)들 전부다 평가 / False일 때, 한장 평가
        # image_path='Dataset/test/2. csm_meng_meng_baby_1_88cad0f74f.jpg',
        image_path='Dataset/test',
        weight_path="weights",
        load_name="480_640_ADAM_PCENTER_RES18",
        load_period=200,
        topk=100,
        nms=False,
        except_class_thresh=0.01,
        nms_thresh=0.5,
        plot_class_thresh=0.5,
        image_save_path="result_image",
        image_show=False,
        image_save=True):

    if mx.context.num_gpus() > 0:
        GPU_COUNT = mx.context.num_gpus()
    else:
        GPU_COUNT = 0

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
    scale_factor = 4  # 고정
    logging.info(f"scale factor {scale_factor}")

    netheight = int(load_name.split("_")[0])
    netwidth = int(load_name.split("_")[1])
    if not isinstance(netheight, int) and not isinstance(netwidth, int):
        logging.info("height is not int")
        logging.info("width is not int")
        raise ValueError
    else:
        logging.info(f"network input size : {(netheight, netwidth)}")

    _, test_dataset = testdataloader()

    weight_path = os.path.join(weight_path, load_name)
    sym = os.path.join(weight_path, f'{load_name}-symbol.json')
    params = os.path.join(weight_path, f'{load_name}-{load_period:04d}.params')

    logging.info("symbol model test")
    if os.path.exists(sym) and os.path.exists(params):
        logging.info(f"loading {os.path.basename(params)} weights\n")
        net = gluon.SymbolBlock.imports(sym,
                                        ['data'],
                                        params, ctx=ctx)
    else:
        raise FileExistsError

    try:
        net = export_block_for_cplusplus(block=net,
                                         data_shape=tuple((netheight, netwidth)) + tuple((3,)),
                                         preprocess=True,  # c++ 에서 inference시 opencv에서 읽은 이미지 그대로 넣으면 됨
                                         layout='HWC',
                                         ctx=ctx)
    except Exception as E:
        logging.error(f"adding preprocessing layer 실패 : {E}")
    else:
        logging.info(f"adding preprocessing layer 성공 ")

    net.hybridize(active=True, static_alloc=True, static_shape=True)

    prediction = Prediction(topk=topk, scale=scale_factor, nms=nms, except_class_thresh=except_class_thresh, nms_thresh=nms_thresh)

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

                heatmap_pred, offset_pred, wh_pred = net(image)
                ids, scores, bboxes = prediction(heatmap_pred, offset_pred, wh_pred)

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

        heatmap_pred, offset_pred, wh_pred = net(image)
        ids, scores, bboxes = prediction(heatmap_pred, offset_pred, wh_pred)

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
        load_name="480_640_ADAM_PCENTER_RES18",
        load_period=200,
        topk=100,
        nms=False,
        except_class_thresh=0.01,
        nms_thresh=0.5,
        plot_class_thresh=0.5,
        image_save_path="result_image",
        image_show=False,
        image_save=True)
