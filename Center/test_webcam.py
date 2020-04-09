import logging
import os
import platform

import cv2
import mxnet as mx
import mxnet.gluon as gluon

from core import Prediction
from core import box_resize
from core import plot_bbox, export_block_for_cplusplus
from core import testdataloader

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def run(weight_path="weights",
        load_name="480_640_ADAM_PCENTER_RES18",
        load_period=1, GPU_COUNT=1,
        topk=500,
        plot_class_thresh=0.5,
        video_name="webcam",
        video_save_path="result_video",
        video_show=True,
        video_save=True):
    if video_save:
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)

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

    try:
        _, test_dataset = testdataloader()

    except Exception:
        logging.info("The dataset does not exist")
        exit(0)

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

    prediction = Prediction(topk=topk, scale=scale_factor)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(os.path.join(video_save_path, f'{video_name}.avi'), fourcc, fps, (width, height))
    logging.info(f"real input size : {(height, width)}")

    while True:
        ret, image = cap.read()
        if ret:
            origin_image = image.copy()
            image = cv2.resize(image, (netwidth, netheight), interpolation=3)
            image[:, :, (0, 1, 2)] = image[:, :, (2, 1, 0)]  # BGR to RGB
            image = mx.nd.array(image, ctx=ctx)
            image = image.expand_dims(axis=0)

            heatmap_pred, offset_pred, wh_pred = net(image)
            ids, scores, bboxes = prediction(heatmap_pred, offset_pred, wh_pred)

            bbox = box_resize(bboxes[0], (netwidth, netheight), (width, height))
            result = plot_bbox(origin_image, bbox, scores=scores[0], labels=ids[0],
                               thresh=plot_class_thresh,
                               reverse_rgb=False,
                               class_names=test_dataset.classes)
            if video_save:
                out.write(result)
            if video_show:
                cv2.imshow(video_name, result)
                cv2.waitKey(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(weight_path="weights",
        load_name="480_640_ADAM_PCENTER_RES18",
        load_period=1, GPU_COUNT=1,
        topk=500,
        plot_class_thresh=0.5,
        video_name="webcam",
        video_save_path="result_video",
        video_show=True,
        video_save=True)
