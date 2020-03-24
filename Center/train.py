import glob
import logging
import os
import platform
import time
from collections import OrderedDict

import cv2
import gluoncv
import mlflow as ml
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.contrib.amp as amp
import mxnet.gluon as gluon
import numpy as np
from mxboard import SummaryWriter
from tqdm import tqdm

from core import CenterNet
from core import HeatmapFocalLoss, NormedL1Loss
from core import Prediction
from core import Voc_2007_AP
from core import plot_bbox, export_block_for_cplusplus, PostNet
from core import traindataloader, validdataloader

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        graphviz=True,
        epoch=100,
        input_size=[512, 512],
        batch_size=16,
        batch_log=100,
        batch_interval=10,
        subdivision=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        multiscale=True,
        factor_scale=[8, 5],
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
        lambda_off=1,
        lambda_size=0.1,
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        GPU_COUNT=0,
        base=18,
        pretrained_base=True,
        pretrained_path="modelparam",
        AMP=True,
        valid_size=8,
        eval_period=5,
        tensorboard=True,
        valid_graph_path="valid_Graph",
        valid_html_auto_open=True,
        using_mlflow=True,
        topk=100,
        iou_thresh=0.5,
        plot_class_thresh=0.5):
    '''
    AMP 가 모든 연산을 지원하지는 않는다.
    modulated convolution을 지원하지 않음
    '''
    if GPU_COUNT == 0:
        ctx = mx.cpu(0)
        AMP = False
    elif GPU_COUNT == 1:
        ctx = mx.gpu(0)
    else:
        ctx = [mx.gpu(i) for i in range(GPU_COUNT)]

    # 운영체제 확인
    if platform.system() == "Linux":
        logging.info(f"{platform.system()} OS")
    elif platform.system() == "Windows":
        logging.info(f"{platform.system()} OS")
    else:
        logging.info(f"{platform.system()} OS")

    if isinstance(ctx, (list, tuple)):
        for i, c in enumerate(ctx):
            free_memory, total_memory = mx.context.gpu_memory_info(i)
            free_memory = round(free_memory / (1024 * 1024 * 1024), 2)
            total_memory = round(total_memory / (1024 * 1024 * 1024), 2)
            logging.info(f'Running on {c} / free memory : {free_memory}GB / total memory {total_memory}GB')
    else:
        if GPU_COUNT == 1:
            free_memory, total_memory = mx.context.gpu_memory_info(0)
            free_memory = round(free_memory / (1024 * 1024 * 1024), 2)
            total_memory = round(total_memory / (1024 * 1024 * 1024), 2)
            logging.info(f'Running on {ctx} / free memory : {free_memory}GB / total memory {total_memory}GB')
        else:
            logging.info(f'Running on {ctx}')

    if GPU_COUNT > 0 and batch_size < GPU_COUNT:
        logging.info("batch size must be greater than gpu number")
        exit(0)

    if AMP:
        amp.init()

    if multiscale:
        logging.info("Using MultiScale")

    if data_augmentation:
        logging.info("Using Data Augmentation")

    logging.info("training Center Detector")
    input_shape = (1, 3) + tuple(input_size)

    scale_factor = 4  # 고정
    logging.info(f"scale factor {scale_factor}")

    try:
        train_dataloader, train_dataset = traindataloader(multiscale=multiscale,
                                                          factor_scale=factor_scale,
                                                          augmentation=data_augmentation,
                                                          path=train_dataset_path,
                                                          input_size=input_size,
                                                          batch_size=batch_size,
                                                          batch_interval=batch_interval,
                                                          num_workers=num_workers,
                                                          shuffle=True, mean=mean, std=std, scale_factor=scale_factor,
                                                          make_target=True)
        valid_dataloader, valid_dataset = validdataloader(path=valid_dataset_path,
                                                          input_size=input_size,
                                                          batch_size=valid_size,
                                                          num_workers=num_workers,
                                                          shuffle=True, mean=mean, std=std, scale_factor=scale_factor,
                                                          make_target=True)

    except Exception:
        logging.info("dataset 없음")
        exit(0)

    train_update_number_per_epoch = len(train_dataloader)
    if train_update_number_per_epoch < 1:
        logging.warning("train batch size가 데이터 수보다 큼")
        exit(0)

    valid_list = glob.glob(os.path.join(valid_dataset_path, "*"))
    if valid_list:
        valid_update_number_per_epoch = len(valid_dataloader)
        if valid_update_number_per_epoch < 1:
            logging.warning("valid batch size가 데이터 수보다 큼")
            exit(0)

    num_classes = train_dataset.num_class  # 클래스 수
    name_classes = train_dataset.classes

    optimizer = optimizer.upper()
    if pretrained_base:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_P" + "CENTER_RES" + str(base)
    else:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_CENTER_RES" + str(base)

    weight_path = f"weights/{model}"
    sym_path = os.path.join(weight_path, f'{model}-symbol.json')
    param_path = os.path.join(weight_path, f'{model}-{load_period:04d}.params')

    if os.path.exists(param_path) and os.path.exists(sym_path):
        start_epoch = load_period
        logging.info(f"loading {os.path.basename(param_path)} weights\n")
        net = gluon.SymbolBlock.imports(sym_path,
                                        ['data'],
                                        param_path, ctx=ctx)
    else:
        start_epoch = 0
        net = CenterNet(base=base,
                        heads=OrderedDict([
                            ('heatmap', {'num_output': num_classes, 'bias': -2.19}),
                            ('offset', {'num_output': 2}),
                            ('wh', {'num_output': 2})
                        ]),
                        head_conv_channel=64,
                        pretrained=pretrained_base,
                        root=pretrained_path,
                        use_dcnv2=False, ctx=ctx)

        if isinstance(ctx, (list, tuple)):
            net.summary(mx.nd.ones(shape=input_shape, ctx=ctx[0]))
        else:
            net.summary(mx.nd.ones(shape=input_shape, ctx=ctx))

        '''
        active (bool, default True) – Whether to turn hybrid on or off.
        static_alloc (bool, default False) – Statically allocate memory to improve speed. Memory usage may increase.
        static_shape (bool, default False) – Optimize for invariant input shapes between iterations. Must also set static_alloc to True. Change of input shapes is still allowed but slower.
        '''
        if multiscale:
            net.hybridize(active=True, static_alloc=True, static_shape=False)
        else:
            net.hybridize(active=True, static_alloc=True, static_shape=True)

    if start_epoch + 1 >= epoch + 1:
        logging.info("this model has already been optimized")
        exit(0)

    if tensorboard:
        summary = SummaryWriter(logdir=os.path.join("mxboard", model), max_queue=10, flush_secs=10,
                                verbose=False)
        if isinstance(ctx, (list, tuple)):
            net.forward(mx.nd.ones(shape=input_shape, ctx=ctx[0]))
        else:
            net.forward(mx.nd.ones(shape=input_shape, ctx=ctx))
        summary.add_graph(net)
    if graphviz:
        gluoncv.utils.viz.plot_network(net, shape=input_shape, save_prefix=model)

    # optimizer
    unit = 1 if (len(train_dataset) // batch_size) < 1 else len(train_dataset) // batch_size
    step = unit * decay_step
    lr_sch = mx.lr_scheduler.FactorScheduler(step=step, factor=decay_lr, stop_factor_lr=1e-12, base_lr=learning_rate)

    for p in net.collect_params().values():
        if p.grad_req != "null":
            p.grad_req = 'add'

    if AMP:
        '''
        update_on_kvstore : bool, default None
        Whether to perform parameter updates on kvstore. If None, then trainer will choose the more
        suitable option depending on the type of kvstore. If the `update_on_kvstore` argument is
        provided, environment variable `MXNET_UPDATE_ON_KVSTORE` will be ignored.
        '''
        if optimizer.upper() == "ADAM":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "beta1": 0.9,
                                                                                       "beta2": 0.999,
                                                                                       'multi_precision': False},
                                    update_on_kvstore=False)  # for Dynamic loss scaling
        elif optimizer.upper() == "RMSPROP":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "gamma1": 0.9,
                                                                                       "gamma2": 0.999,
                                                                                       'multi_precision': False},
                                    update_on_kvstore=False)  # for Dynamic loss scaling
        elif optimizer.upper() == "SGD":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "wd": 0.0001,
                                                                                       "momentum": 0.9,
                                                                                       'multi_precision': False},
                                    update_on_kvstore=False)  # for Dynamic loss scaling
        else:
            logging.error("optimizer not selected")
            exit(0)

        amp.init_trainer(trainer)

    else:
        if optimizer.upper() == "ADAM":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "beta1": 0.9,
                                                                                       "beta2": 0.999,
                                                                                       'multi_precision': False})
        elif optimizer.upper() == "RMSPROP":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "gamma1": 0.9,
                                                                                       "gamma2": 0.999,
                                                                                       'multi_precision': False})
        elif optimizer.upper() == "SGD":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "wd": 0.0001,
                                                                                       "momentum": 0.9,
                                                                                       'multi_precision': False})

        else:
            logging.error("optimizer not selected")
            exit(0)

    heatmapfocalloss = HeatmapFocalLoss(from_sigmoid=True, alpha=2, beta=4)
    normedl1loss = NormedL1Loss()
    prediction = Prediction(batch_size=valid_size, topk=topk, scale=scale_factor)
    precision_recall = Voc_2007_AP(iou_thresh=iou_thresh, class_names=name_classes)

    start_time = time.time()
    for i in tqdm(range(start_epoch + 1, epoch + 1, 1), initial=start_epoch + 1, total=epoch):

        heatmap_loss_sum = 0
        offset_loss_sum = 0
        wh_loss_sum = 0
        time_stamp = time.time()

        '''
        target generator를 train_dataloader에서 만들어 버리는게 학습 속도가 훨씬 빠르다. 
        '''

        for batch_count, (image, _, heatmap, offset_target, wh_target, mask_target, _) in enumerate(
                train_dataloader,
                start=1):
            td_batch_size = image.shape[0]

            image_split = mx.nd.split(data=image, num_outputs=subdivision, axis=0)
            heatmap_split = mx.nd.split(data=heatmap, num_outputs=subdivision, axis=0)
            offset_target_split = mx.nd.split(data=offset_target, num_outputs=subdivision, axis=0)
            wh_target_split = mx.nd.split(data=wh_target, num_outputs=subdivision, axis=0)
            mask_target_split = mx.nd.split(data=mask_target, num_outputs=subdivision, axis=0)

            if subdivision == 1:
                image_split = [image_split]
                heatmap_split = [heatmap_split]
                offset_target_split = [offset_target_split]
                wh_target_split = [wh_target_split]
                mask_target_split = [mask_target_split]

            '''
            autograd 설명
            https://mxnet.apache.org/api/python/docs/tutorials/getting-started/crash-course/3-autograd.html
            '''
            with autograd.record(train_mode=True):

                heatmap_all_losses = []
                offset_all_losses = []
                wh_all_losses = []

                for image_part, heatmap_part, offset_target_part, wh_target_part, mask_target_part in zip(image_split,
                                                                                                          heatmap_split,
                                                                                                          offset_target_split,
                                                                                                          wh_target_split,
                                                                                                          mask_target_split):

                    if GPU_COUNT <= 1:
                        image_part = gluon.utils.split_and_load(image_part, [ctx], even_split=False)
                        heatmap_part = gluon.utils.split_and_load(heatmap_part, [ctx], even_split=False)
                        offset_target_part = gluon.utils.split_and_load(offset_target_part, [ctx], even_split=False)
                        wh_target_part = gluon.utils.split_and_load(wh_target_part, [ctx], even_split=False)
                        mask_target_part = gluon.utils.split_and_load(mask_target_part, [ctx], even_split=False)
                    else:
                        image_part = gluon.utils.split_and_load(image_part, ctx, even_split=False)
                        heatmap_part = gluon.utils.split_and_load(heatmap_part, ctx, even_split=False)
                        offset_target_part = gluon.utils.split_and_load(offset_target_part, ctx, even_split=False)
                        wh_target_part = gluon.utils.split_and_load(wh_target_part, ctx, even_split=False)
                        mask_target_part = gluon.utils.split_and_load(mask_target_part, ctx, even_split=False)

                    # prediction, target space for Data Parallelism
                    heatmap_losses = []
                    offset_losses = []
                    wh_losses = []
                    total_loss = []

                    # gpu N 개를 대비한 코드 (Data Parallelism)
                    for img, heatmap_target, offset_target, wh_target, mask_target in zip(image_part, heatmap_part,
                                                                                          offset_target_part,
                                                                                          wh_target_part,
                                                                                          mask_target_part):
                        heatmap_pred, offset_pred, wh_pred = net(img)
                        heatmap_loss = heatmapfocalloss(heatmap_pred, heatmap_target)
                        offset_loss = normedl1loss(offset_pred, offset_target, mask_target) * lambda_off
                        wh_loss = normedl1loss(wh_pred, wh_target, mask_target) * lambda_size

                        heatmap_losses.append(heatmap_loss.asscalar())
                        offset_losses.append(offset_loss.asscalar())
                        wh_losses.append(wh_loss.asscalar())

                        total_loss.append(heatmap_loss + offset_loss + wh_loss)

                    if AMP:
                        with amp.scale_loss(total_loss, trainer) as scaled_loss:
                            autograd.backward(scaled_loss)
                    else:
                        autograd.backward(total_loss)

                    heatmap_all_losses.append(sum(heatmap_losses))
                    offset_all_losses.append(sum(offset_losses))
                    wh_all_losses.append(sum(wh_losses))

            trainer.step(batch_size=td_batch_size, ignore_stale_grad=False)
            # 비우기

            for p in net.collect_params().values():
                p.zero_grad()

            heatmap_loss_sum += sum(heatmap_all_losses) / td_batch_size
            offset_loss_sum += sum(offset_all_losses) / td_batch_size
            wh_loss_sum += sum(wh_all_losses) / td_batch_size

            if batch_count % batch_log == 0:
                logging.info(f'[Epoch {i}][Batch {batch_count}/{train_update_number_per_epoch}],'
                             f'[Speed {td_batch_size / (time.time() - time_stamp):.3f} samples/sec],'
                             f'[Lr = {trainer.learning_rate}]'
                             f'[heatmap loss = {sum(heatmap_all_losses) / td_batch_size:.3f}]'
                             f'[offset loss = {sum(offset_all_losses) / td_batch_size:.3f}]'
                             f'[wh loss = {sum(wh_all_losses) / td_batch_size:.3f}]')
            time_stamp = time.time()

        train_heatmap_loss_mean = np.divide(heatmap_loss_sum, train_update_number_per_epoch)
        train_offset_loss_mean = np.divide(offset_loss_sum, train_update_number_per_epoch)
        train_wh_loss_mean = np.divide(wh_loss_sum, train_update_number_per_epoch)
        train_total_loss_mean = train_heatmap_loss_mean + train_offset_loss_mean + train_wh_loss_mean

        logging.info(
            f"train heatmap loss : {train_heatmap_loss_mean} / train offset loss : {train_offset_loss_mean} / train wh loss : {train_wh_loss_mean} / train total loss : {train_total_loss_mean}")

        if i % eval_period == 0 and valid_list:

            heatmap_loss_sum = 0
            offset_loss_sum = 0
            wh_loss_sum = 0

            # loss 구하기
            for image, label, heatmap_all, offset_target_all, wh_target_all, mask_target_all, _ in valid_dataloader:
                vd_batch_size = image.shape[0]

                if GPU_COUNT <= 1:
                    image = gluon.utils.split_and_load(image, [ctx], even_split=False)
                    label = gluon.utils.split_and_load(label, [ctx], even_split=False)
                    heatmap_split = gluon.utils.split_and_load(heatmap_all, [ctx], even_split=False)
                    offset_target_split = gluon.utils.split_and_load(offset_target_all, [ctx], even_split=False)
                    wh_target_split = gluon.utils.split_and_load(wh_target_all, [ctx], even_split=False)
                    mask_target_split = gluon.utils.split_and_load(mask_target_all, [ctx], even_split=False)
                else:
                    image = gluon.utils.split_and_load(image, ctx, even_split=False)
                    label = gluon.utils.split_and_load(label, ctx, even_split=False)
                    heatmap_split = gluon.utils.split_and_load(heatmap_all, ctx, even_split=False)
                    offset_target_split = gluon.utils.split_and_load(offset_target_all, ctx, even_split=False)
                    wh_target_split = gluon.utils.split_and_load(wh_target_all, ctx, even_split=False)
                    mask_target_split = gluon.utils.split_and_load(mask_target_all, ctx, even_split=False)

                # prediction, target space for Data Parallelism
                heatmap_losses = []
                offset_losses = []
                wh_losses = []

                # gpu N 개를 대비한 코드 (Data Parallelism)
                for img, lb, heatmap_target, offset_target, wh_target, mask_target in zip(image, label, heatmap_split,
                                                                                          offset_target_split,
                                                                                          wh_target_split,
                                                                                          mask_target_split):
                    gt_box = lb[:, :, :4]
                    gt_id = lb[:, :, 4:5]
                    heatmap_pred, offset_pred, wh_pred = net(img)

                    id, score, bbox = prediction(heatmap_pred, offset_pred, wh_pred)
                    precision_recall.update(pred_bboxes=bbox,
                                            pred_labels=id,
                                            pred_scores=score,
                                            gt_boxes=gt_box * scale_factor,
                                            gt_labels=gt_id)

                    heatmap_loss = heatmapfocalloss(heatmap_pred, heatmap_target)
                    offset_loss = normedl1loss(offset_pred, offset_target, mask_target) * lambda_off
                    wh_loss = normedl1loss(wh_pred, wh_target, mask_target) * lambda_size

                    heatmap_losses.append(heatmap_loss.asscalar())
                    offset_losses.append(offset_loss.asscalar())
                    wh_losses.append(wh_loss.asscalar())

                heatmap_loss_sum += sum(heatmap_losses) / vd_batch_size
                offset_loss_sum += sum(offset_losses) / vd_batch_size
                wh_loss_sum += sum(wh_losses) / vd_batch_size

            valid_heatmap_loss_mean = np.divide(heatmap_loss_sum, valid_update_number_per_epoch)
            valid_offset_loss_mean = np.divide(offset_loss_sum, valid_update_number_per_epoch)
            valid_wh_loss_mean = np.divide(wh_loss_sum, valid_update_number_per_epoch)
            valid_total_loss_mean = valid_heatmap_loss_mean + valid_offset_loss_mean + valid_wh_loss_mean

            logging.info(
                f"valid heatmap loss : {valid_heatmap_loss_mean} / valid offset loss : {valid_offset_loss_mean} / valid wh loss : {valid_wh_loss_mean} / valid total loss : {valid_total_loss_mean}")

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
                                          AP=AP_appender, mAP=mAP_result, folder_name=valid_graph_path, epoch=i,
                                          auto_open=valid_html_auto_open)
            precision_recall.reset()

            if tensorboard:
                # gpu N 개를 대비한 코드 (Data Parallelism)
                dataloader_iter = iter(valid_dataloader)
                image, label, _, _, _, _, _ = next(dataloader_iter)

                if GPU_COUNT <= 1:
                    image = gluon.utils.split_and_load(image, [ctx], even_split=False)
                    label = gluon.utils.split_and_load(label, [ctx], even_split=False)
                else:
                    image = gluon.utils.split_and_load(image, ctx, even_split=False)
                    label = gluon.utils.split_and_load(label, ctx, even_split=False)

                ground_truth_colors = {}
                for k in range(num_classes):
                    ground_truth_colors[k] = (0, 0, 1)

                batch_image = []
                heatmap_image = []
                for img, lb in zip(image, label):
                    gt_boxes = lb[:, :, :4]
                    gt_ids = lb[:, :, 4:5]
                    heatmap_pred, offset_pred, wh_pred = net(img)
                    ids, scores, bboxes = prediction(heatmap_pred, offset_pred, wh_pred)

                    for ig, gt_id, gt_box, heatmap, id, score, bbox in zip(img, gt_ids, gt_boxes, heatmap_pred, ids,
                                                                           scores, bboxes):
                        ig = ig.transpose(
                            (1, 2, 0)) * mx.nd.array(std, ctx=ig.context) + mx.nd.array(mean, ctx=ig.context)
                        ig = (ig * 255).clip(0, 255)

                        # heatmap 그리기
                        heatmap = mx.nd.multiply(heatmap, 255.0)  # 0 ~ 255 범위로 바꾸기
                        heatmap = mx.nd.max(heatmap, axis=0, keepdims=True)  # channel 축으로 가장 큰것 뽑기
                        heatmap = mx.nd.transpose(heatmap, axes=(1, 2, 0))  # (height, width, channel=1)
                        heatmap = mx.nd.repeat(heatmap, repeats=3, axis=-1)  # (height, width, channel=3)
                        heatmap = heatmap.asnumpy()  # mxnet.ndarray -> numpy.ndarray
                        heatmap = cv2.resize(heatmap, dsize=(input_size[1], input_size[0]))  # 사이즈 원복
                        heatmap = heatmap.astype("uint8")  # float32 -> uint8
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        heatmap[:, :, (0, 1, 2)] = heatmap[:, :, (2, 1, 0)]  # BGR -> RGB
                        heatmap = np.transpose(heatmap, axes=(2, 0, 1))  # (channel=3, height, width)

                        # ground truth box 그리기
                        ground_truth = plot_bbox(ig, gt_box * scale_factor, scores=None, labels=gt_id, thresh=None,
                                                 reverse_rgb=True,
                                                 class_names=valid_dataset.classes, absolute_coordinates=True,
                                                 colors=ground_truth_colors)
                        # prediction box 그리기
                        prediction_box = plot_bbox(ground_truth, bbox, scores=score, labels=id,
                                                   thresh=plot_class_thresh,
                                                   reverse_rgb=False,
                                                   class_names=valid_dataset.classes, absolute_coordinates=True)

                        # Tensorboard에 그리기 위해 BGR -> RGB / (height, width, channel) -> (channel, height, width) 를한다.
                        prediction_box = cv2.cvtColor(prediction_box, cv2.COLOR_BGR2RGB)
                        prediction_box = np.transpose(prediction_box,
                                                      axes=(2, 0, 1))
                        batch_image.append(prediction_box)  # (batch, channel, height, width)
                        heatmap_image.append(heatmap)

                all_image = np.concatenate([np.array(batch_image), np.array(heatmap_image)], axis=-1)
                summary.add_image(tag="valid_result", image=all_image, global_step=i)
                summary.add_scalar(tag="heatmap_loss", value={"train_heatmap_loss_mean": train_heatmap_loss_mean,
                                                              "valid_heatmap_loss_mean": valid_heatmap_loss_mean},
                                   global_step=i)
                summary.add_scalar(tag="offset_loss",
                                   value={"train_offset_loss_mean": train_offset_loss_mean,
                                          "valid_offset_loss_mean": valid_offset_loss_mean},
                                   global_step=i)
                summary.add_scalar(tag="wh_loss",
                                   value={"train_wh_loss_mean": train_wh_loss_mean,
                                          "valid_wh_loss_mean": valid_wh_loss_mean},
                                   global_step=i)

                summary.add_scalar(tag="total_loss", value={
                    "train_total_loss": train_total_loss_mean,
                    "valid_total_loss": valid_total_loss_mean},
                                   global_step=i)

                params = net.collect_params().values()
                if GPU_COUNT > 1:
                    for c in ctx:
                        for p in params:
                            summary.add_histogram(tag=p.name, values=p.data(ctx=c), global_step=i, bins='default')
                else:
                    for p in params:
                        summary.add_histogram(tag=p.name, values=p.data(), global_step=i, bins='default')

        if i % save_period == 0:

            weight_epoch_path = os.path.join(weight_path, str(i))
            if not os.path.exists(weight_epoch_path):
                os.makedirs(weight_epoch_path)
            '''
            Hybrid models can be serialized as JSON files using the export function
            Export HybridBlock to json format that can be loaded by SymbolBlock.imports, mxnet.mod.Module or the C++ interface.
            When there are only one input, it will have name data. When there Are more than one inputs, they will be named as data0, data1, etc.
            '''
            if GPU_COUNT >= 1:
                context = mx.gpu(0)
            else:
                context = mx.cpu(0)

            postnet = PostNet(net=net, auxnet=prediction)  # 새로운 객체가 생성
            try:
                net.export(os.path.join(weight_path, f"{model}"), epoch=i, remove_amp_cast=True)
                net.save_parameters(os.path.join(weight_path, f"{i}.params"))  # onnx 추출용
                # network inference, decoder, nms까지 처리됨 - mxnet c++에서 편리함
                export_block_for_cplusplus(path=os.path.join(weight_epoch_path, f"{model}_prepost"),
                                           block=postnet,
                                           data_shape=tuple(input_size) + tuple((3,)),
                                           epoch=i,
                                           preprocess=True,  # c++ 에서 inference시 opencv에서 읽은 이미지 그대로 넣으면 됨
                                           layout='HWC',
                                           ctx=context,
                                           remove_amp_cast=True)

            except Exception as E:
                logging.error(f"json, param model export 예외 발생 : {E}")
            else:
                logging.info("json, param model export 성공")
                net.collect_params().reset_ctx(ctx)

    end_time = time.time()
    learning_time = end_time - start_time
    logging.info(f"learning time : 약, {learning_time / 3600:0.2f}H")
    logging.info("optimization completed")

    if using_mlflow:
        ml.log_metric("learning time", round(learning_time / 3600, 2))


if __name__ == "__main__":
    run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        graphviz=True,
        epoch=100,
        input_size=[512, 512],
        batch_size=16,
        batch_log=100,
        batch_interval=10,
        subdivision=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        multiscale=True,
        factor_scale=[8, 5],
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
        lambda_off=1,
        lambda_size=0.1,
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        GPU_COUNT=0,
        base=18,
        pretrained_base=True,
        pretrained_path="modelparam",
        AMP=True,
        valid_size=8,
        eval_period=5,
        tensorboard=True,
        valid_graph_path="valid_Graph",
        valid_html_auto_open=True,
        using_mlflow=True,
        topk=100,
        iou_thresh=0.5,
        plot_class_thresh=0.5)
