import logging
import os
import platform

import cv2
import mxnet as mx
import mxnet.gluon as gluon
import numpy as np
from tqdm import tqdm

from core import HeatmapFocalLoss, NormedL1Loss
from core import TargetGenerator, Prediction
from core import Voc_2007_AP
from core import plot_bbox, box_resize
from core import testdataloader

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        load_name="480_640_ADAM_PCENTER_RES18", load_period=10, GPU_COUNT=0,
        test_weight_path="weights",
        test_dataset_path="Dataset/test",
        test_save_path="result",
        test_graph_path="test_Graph",
        lambda_off=1,
        lambda_size=0.1,
        num_workers=4,
        show_flag=True,
        save_flag=True,
        topk=100,
        plot_class_thresh=0.5):
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
        test_dataloader, test_dataset = testdataloader(path=test_dataset_path,
                                                       input_size=(netheight, netwidth),
                                                       num_workers=num_workers,
                                                       mean=mean, std=std, scale_factor=scale_factor)
    except Exception:
        logging.info("The dataset does not exist")
        exit(0)

    weight_path = os.path.join(test_weight_path, load_name)
    sym = os.path.join(weight_path, f'{load_name}-symbol.json')
    params = os.path.join(weight_path, f'{load_name}-{load_period:04d}.params')

    test_update_number_per_epoch = len(test_dataloader)
    if test_update_number_per_epoch < 1:
        logging.warning(" test batch size가 데이터 수보다 큼 ")
        exit(0)

    num_classes = test_dataset.num_class  # 클래스 수
    name_classes = test_dataset.classes

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

    heatmapfocalloss = HeatmapFocalLoss(from_sigmoid=True, alpha=2, beta=4)
    normedl1loss = NormedL1Loss()
    targetgenerator = TargetGenerator(num_classes=num_classes)
    prediction = Prediction(topk=topk, scale=scale_factor)

    precision_recall = Voc_2007_AP(iou_thresh=0.5, class_names=name_classes)

    ground_truth_colors = {}
    for i in range(num_classes):
        ground_truth_colors[i] = (0, 0, 1)

    heatmap_loss_sum = 0
    offset_loss_sum = 0
    wh_loss_sum = 0

    for image, label, origin_image, origin_box, name in tqdm(test_dataloader):
        _, height, width, _ = origin_image.shape
        logging.info(f"real input size : {(height, width)}")
        origin_image = origin_image.asnumpy()[0]
        origin_box = origin_box.asnumpy()[0]

        image = image.as_in_context(ctx)
        label = label.as_in_context(ctx)
        gt_boxes = label[:, :, :4]
        gt_ids = label[:, :, 4:5]
        heatmap_pred, offset_pred, wh_pred = net(image)
        ids, scores, bboxes = prediction(heatmap_pred, offset_pred, wh_pred)

        precision_recall.update(pred_bboxes=bboxes,
                                pred_labels=ids,
                                pred_scores=scores,
                                gt_boxes=gt_boxes * scale_factor,
                                gt_labels=gt_ids)

        heatmap = mx.nd.multiply(heatmap_pred[0], 255.0)  # 0 ~ 255 범위로 바꾸기
        heatmap = mx.nd.max(heatmap, axis=0, keepdims=True)  # channel 축으로 가장 큰것 뽑기
        heatmap = mx.nd.transpose(heatmap, axes=(1, 2, 0))  # (height, width, channel=1)
        heatmap = mx.nd.repeat(heatmap, repeats=3, axis=-1)  # (height, width, channel=3)
        heatmap = heatmap.asnumpy()  # mxnet.ndarray -> numpy.ndarray
        heatmap = cv2.resize(heatmap, dsize=(width, height))  # 사이즈 원복
        heatmap = heatmap.astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # heatmap, image add하기
        bbox = box_resize(bboxes[0], (netwidth, netheight), (width, height))
        ground_truth = plot_bbox(origin_image, origin_box[:, :4], scores=None, labels=origin_box[:, 4:5], thresh=None,
                                 reverse_rgb=True,
                                 class_names=test_dataset.classes, absolute_coordinates=True,
                                 colors=ground_truth_colors)
        plot_bbox(ground_truth, bbox, scores=scores[0], labels=ids[0], thresh=plot_class_thresh,
                  reverse_rgb=False,
                  class_names=test_dataset.classes, absolute_coordinates=True,
                  image_show=show_flag, image_save=save_flag, image_save_path=test_save_path, image_name=name[0], heatmap=heatmap)

        heatmap_target, offset_target, wh_target, mask_target = targetgenerator(gt_boxes, gt_ids,
                                                                                netwidth // scale_factor, netheight // scale_factor, image.context)
        heatmap_loss = heatmapfocalloss(heatmap_pred, heatmap_target)
        offset_loss = normedl1loss(offset_pred, offset_target, mask_target) * lambda_off
        wh_loss = normedl1loss(wh_pred, wh_target, mask_target) * lambda_size

        heatmap_loss_sum += heatmap_loss.asscalar()
        offset_loss_sum += offset_loss.asscalar()
        wh_loss_sum += wh_loss.asscalar()

    # epoch 당 평균 loss
    test_heatmap_loss_mean = np.divide(heatmap_loss_sum, test_update_number_per_epoch)
    test_offset_loss_mean = np.divide(offset_loss_sum, test_update_number_per_epoch)
    test_wh_loss_mean = np.divide(wh_loss_sum, test_update_number_per_epoch)

    logging.info(
        f"test heatmap loss : {test_heatmap_loss_mean} / test offset loss : {test_offset_loss_mean} / test wh loss : {test_wh_loss_mean}")

    AP_appender = []
    round_position = 2
    class_name, precision, recall, true_positive, false_positive, threshold = precision_recall.get_PR_list()
    for j, c, p, r in zip(range(len(recall)), class_name, precision, recall):
        name, AP = precision_recall.get_AP(c, p, r)
        logging.info(f"class {j}'s {name} AP : {round(AP * 100, round_position)}%")
        AP_appender.append(AP)
    mAP_result = np.mean(AP_appender)

    logging.info(f"mAP : {round(mAP_result * 100, round_position)}%")
    precision_recall.get_PR_curve(name=class_name,
                                  precision=precision,
                                  recall=recall,
                                  threshold=threshold,
                                  AP=AP_appender, mAP=mAP_result, folder_name=test_graph_path)


if __name__ == "__main__":
    run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        load_name="480_640_ADAM_PCENTER_RES18", load_period=10, GPU_COUNT=0,
        test_weight_path="weights",
        test_dataset_path="Dataset/test",
        test_save_path="result",
        test_graph_path="test_Graph",
        lambda_off=1,
        lambda_size=0.1,
        num_workers=4,
        show_flag=True,
        save_flag=True,
        topk=100,
        plot_class_thresh=0.5)
