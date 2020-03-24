import logging
import os
import platform

import mxnet as mx
import mxnet.gluon as gluon
import numpy as np
from core import FocalLoss, HuberLoss
from core import TargetGenerator, Prediction
from core import Voc_2007_AP
from core import box_resize
from core import plot_bbox
from core import testdataloader
from tqdm import tqdm

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        load_name="512_512_ADAM_PEFF_0", load_period=10, GPU_COUNT=0,
        test_weight_path="weights",
        test_dataset_path="Dataset/test",
        test_save_path="result",
        test_graph_path="test_Graph",
        test_html_auto_open=False,
        foreground_iou_thresh=0.5,
        background_iou_thresh=0.4,
        num_workers=4,
        show_flag=True,
        save_flag=True,
        decode_number=5000,
        multiperclass=True,
        nms_thresh=0.5,
        nms_topk=500,
        iou_thresh=0.5,
        except_class_thresh=0.05,
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
                                                       mean=mean, std=std)
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

    confidence_loss = FocalLoss(alpha=0.25,  # 논문에서 가장 좋다고 한 숫자
                                gamma=1.5,  # 논문에서 가장 좋다고 한 숫자
                                sparse_label=True,
                                from_sigmoid=False,
                                batch_axis=None,
                                num_class=num_classes,
                                reduction="sum",
                                exclude=False)

    localization_loss = HuberLoss(rho=1,
                                  batch_axis=None,
                                  reduction="sum",
                                  exclude=False)

    targetgenerator = TargetGenerator(foreground_iou_thresh=foreground_iou_thresh,
                                      background_iou_thresh=background_iou_thresh)

    prediction = Prediction(
        from_sigmoid=False,
        num_classes=num_classes,
        decode_number=decode_number,
        nms_thresh=nms_thresh,
        nms_topk=nms_topk,
        except_class_thresh=except_class_thresh,
        multiperclass=multiperclass)

    precision_recall = Voc_2007_AP(iou_thresh=iou_thresh, class_names=name_classes)

    ground_truth_colors = {}
    for i in range(num_classes):
        ground_truth_colors[i] = (0, 0, 1)

    conf_loss_sum = 0
    loc_loss_sum = 0

    for image, label, name, origin_image, origin_box in tqdm(test_dataloader):
        _, height, width, _ = origin_image.shape
        logging.info(f"real input size : {(height, width)}")
        origin_image = origin_image.asnumpy()[0]
        origin_box = origin_box.asnumpy()[0]

        image = image.as_in_context(ctx)
        label = label.as_in_context(ctx)
        gt_boxes = label[:, :, :4]
        gt_ids = label[:, :, 4:5]

        cls_preds, box_preds, anchors = net(image)
        ids, scores, bboxes = prediction(cls_preds, box_preds, anchors)

        precision_recall.update(pred_bboxes=bboxes,
                                pred_labels=ids,
                                pred_scores=scores,
                                gt_boxes=gt_boxes,
                                gt_labels=gt_ids)

        bbox = box_resize(bboxes[0], (netwidth, netheight), (width, height))
        ground_truth = plot_bbox(origin_image, origin_box[:, :4], scores=None, labels=origin_box[:, 4:5], thresh=None,
                                 reverse_rgb=True,
                                 class_names=test_dataset.classes, absolute_coordinates=True,
                                 colors=ground_truth_colors)
        plot_bbox(ground_truth, bbox, scores=scores[0], labels=ids[0], thresh=plot_class_thresh,
                  reverse_rgb=False,
                  class_names=test_dataset.classes, absolute_coordinates=True,
                  image_show=show_flag, image_save=save_flag, image_save_path=test_save_path, image_name=name[0])

        cls_targets, box_targets = targetgenerator(anchors, gt_boxes, gt_ids)
        except_ignore_samples = cls_targets > -1
        positive_samples = cls_targets > 0
        positive_numbers = positive_samples.sum()

        conf_loss = confidence_loss(cls_preds, cls_targets, except_ignore_samples.expand_dims(axis=-1))
        conf_loss = mx.nd.divide(conf_loss, positive_numbers + 1)
        conf_loss_sum += conf_loss.asscalar()

        loc_loss = localization_loss(box_preds, box_targets, positive_samples.expand_dims(axis=-1))
        loc_loss_sum += loc_loss.asscalar()

    # epoch 당 평균 loss
    test_conf_loss_mean = np.divide(conf_loss_sum, test_update_number_per_epoch)
    test_loc_loss_mean = np.divide(loc_loss_sum, test_update_number_per_epoch)
    test_total_loss_mean = test_conf_loss_mean + test_loc_loss_mean

    logging.info(
        f"test confidence loss : {test_conf_loss_mean} / test localization loss : {test_loc_loss_mean} / test total loss : {test_total_loss_mean}")

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
                                  AP=AP_appender, mAP=mAP_result, folder_name=test_graph_path, epoch=load_period,
                                  auto_open=test_html_auto_open)


if __name__ == "__main__":
    run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        load_name="512_512_ADAM_PEFF_0", load_period=100, GPU_COUNT=0,
        test_weight_path="weights",
        test_dataset_path="Dataset/test",
        test_save_path="result",
        test_graph_path="test_Graph",
        test_html_auto_open=True,
        foreground_iou_thresh=0.5,
        background_iou_thresh=0.4,
        num_workers=4,
        show_flag=True,
        save_flag=True,
        decode_number=5000,
        multiperclass=True,
        nms_thresh=0.5,
        nms_topk=500,
        iou_thresh=0.5,
        except_class_thresh=0.05,
        plot_class_thresh=0.5)
