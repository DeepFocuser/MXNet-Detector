import glob
import logging
import os
import platform
import time

import cv2
import gluoncv
import mlflow as ml
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.contrib.amp as amp
import mxnet.gluon as gluon
import numpy as np
from core import Voc_2007_AP
from core import Yolov3, Yolov3Loss, Prediction
from core import plot_bbox, export_block_for_cplusplus, PostNet
from core import traindataloader, validdataloader
from mxboard import SummaryWriter
from tqdm import tqdm

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        offset_alloc_size=(64, 64),
        anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                 "middle": [(30, 61), (62, 45), (59, 119)],
                 "deep": [(116, 90), (156, 198), (373, 326)]},
        graphviz=False,
        epoch=100,
        input_size=[416, 416],
        batch_log=100,
        batch_size=16,
        batch_interval=10,
        subdivision=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        multiscale=False,
        factor_scale=[13, 5],
        ignore_threshold=0.5,
        dynamic=False,
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        GPU_COUNT=0,
        Darknetlayer=53,
        pretrained_base=True,
        pretrained_path="modelparam",
        AMP=True,
        valid_size=8,
        eval_period=5,
        tensorboard=True,
        valid_graph_path="valid_Graph",
        valid_html_auto_open=True,
        using_mlflow=True,
        multiperclass=True,
        nms_thresh=0.5,
        nms_topk=500,
        iou_thresh=0.5,
        except_class_thresh=0.05,
        plot_class_thresh=0.5):
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

    # 입력 사이즈를 32의 배수로 지정해 버리기 - stride가 일그러지는 것을 막기 위함
    if input_size[0] % 32 != 0 and input_size[1] % 32 != 0:
        logging.info("The input size must be a multiple of 32")
        exit(0)

    if GPU_COUNT > 0 and batch_size < GPU_COUNT:
        logging.info("batch size must be greater than gpu number")
        exit(0)

    if AMP:
        amp.init()

    if multiscale:
        logging.info("Using MultiScale")

    if data_augmentation:
        logging.info("Using Data Augmentation")

    logging.info("training YoloV3 Detector")
    input_shape = (1, 3) + tuple(input_size)

    try:
        net = Yolov3(Darknetlayer=Darknetlayer,
                     anchors=anchors,
                     pretrained=False,
                     ctx=mx.cpu())
        train_dataloader, train_dataset = traindataloader(multiscale=multiscale,
                                                          factor_scale=factor_scale,
                                                          augmentation=data_augmentation,
                                                          path=train_dataset_path,
                                                          input_size=input_size,
                                                          batch_size=batch_size,
                                                          batch_interval=batch_interval,
                                                          num_workers=num_workers,
                                                          shuffle=True, mean=mean, std=std,
                                                          net=net, ignore_threshold=ignore_threshold, dynamic=dynamic,
                                                          from_sigmoid=False, make_target=True)
        valid_dataloader, valid_dataset = validdataloader(path=valid_dataset_path,
                                                          input_size=input_size,
                                                          batch_size=valid_size,
                                                          num_workers=num_workers,
                                                          shuffle=True, mean=mean, std=std,
                                                          net=net, ignore_threshold=ignore_threshold, dynamic=dynamic,
                                                          from_sigmoid=False, make_target=True)

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
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_P" + "Dark_" + str(Darknetlayer)
    else:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_Dark_" + str(Darknetlayer)

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
        '''
        mxnet c++에서 arbitrary input image 를 받기 위한 전략
        alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `alloc_size` to generate large enough offset
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
        '''
        net = Yolov3(Darknetlayer=Darknetlayer,
                     input_size=input_size,
                     anchors=anchors,
                     num_classes=num_classes,  # foreground만
                     pretrained=pretrained_base,
                     pretrained_path=pretrained_path,
                     alloc_size=offset_alloc_size,
                     ctx=ctx)

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
                                                                                       "wd": 0.0005,
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
                                                                                       "wd": 0.0005,
                                                                                       "momentum": 0.9,
                                                                                       'multi_precision': False})

        else:
            logging.error("optimizer not selected")
            exit(0)

    loss = Yolov3Loss(sparse_label=True,
                      from_sigmoid=False,
                      batch_axis=None,
                      num_classes=num_classes,
                      reduction="sum",
                      exclude=False)

    prediction = Prediction(
        from_sigmoid=False,
        num_classes=num_classes,
        nms_thresh=nms_thresh,
        nms_topk=nms_topk,
        except_class_thresh=except_class_thresh,
        multiperclass=multiperclass)

    precision_recall = Voc_2007_AP(iou_thresh=iou_thresh, class_names=name_classes)

    start_time = time.time()
    for i in tqdm(range(start_epoch + 1, epoch + 1, 1), initial=start_epoch + 1, total=epoch):

        xcyc_loss_sum = 0
        wh_loss_sum = 0
        object_loss_sum = 0
        class_loss_sum = 0
        time_stamp = time.time()

        for batch_count, (image, _, xcyc_all, wh_all, objectness_all, class_all, weights_all, _) in enumerate(
                train_dataloader, start=1):
            td_batch_size = image.shape[0]

            image = mx.nd.split(data=image, num_outputs=subdivision, axis=0)
            xcyc_all = mx.nd.split(data=xcyc_all, num_outputs=subdivision, axis=0)
            wh_all = mx.nd.split(data=wh_all, num_outputs=subdivision, axis=0)
            objectness_all = mx.nd.split(data=objectness_all, num_outputs=subdivision, axis=0)
            class_all = mx.nd.split(data=class_all, num_outputs=subdivision, axis=0)
            weights_all = mx.nd.split(data=weights_all, num_outputs=subdivision, axis=0)

            if subdivision == 1:
                image = [image]
                xcyc_all = [xcyc_all]
                wh_all = [wh_all]
                objectness_all = [objectness_all]
                class_all = [class_all]
                weights_all = [weights_all]
            '''
            autograd 설명
            https://mxnet.apache.org/api/python/docs/tutorials/getting-started/crash-course/3-autograd.html
            '''
            with autograd.record(train_mode=True):

                xcyc_all_losses = []
                wh_all_losses = []
                object_all_losses = []
                class_all_losses = []

                for image_split, xcyc_split, wh_split, objectness_split, class_split, weights_split in zip(image,
                                                                                                           xcyc_all,
                                                                                                           wh_all,
                                                                                                           objectness_all,
                                                                                                           class_all,
                                                                                                           weights_all):

                    if GPU_COUNT <= 1:
                        image_split = gluon.utils.split_and_load(image_split, [ctx], even_split=False)
                        xcyc_split = gluon.utils.split_and_load(xcyc_split, [ctx], even_split=False)
                        wh_split = gluon.utils.split_and_load(wh_split, [ctx], even_split=False)
                        objectness_split = gluon.utils.split_and_load(objectness_split, [ctx], even_split=False)
                        class_split = gluon.utils.split_and_load(class_split, [ctx], even_split=False)
                        weights_split = gluon.utils.split_and_load(weights_split, [ctx], even_split=False)
                    else:
                        image_split = gluon.utils.split_and_load(image_split, ctx, even_split=False)
                        xcyc_split = gluon.utils.split_and_load(xcyc_split, ctx, even_split=False)
                        wh_split = gluon.utils.split_and_load(wh_split, ctx, even_split=False)
                        objectness_split = gluon.utils.split_and_load(objectness_split, ctx, even_split=False)
                        class_split = gluon.utils.split_and_load(class_split, ctx, even_split=False)
                        weights_split = gluon.utils.split_and_load(weights_split, ctx, even_split=False)

                    xcyc_losses = []
                    wh_losses = []
                    object_losses = []
                    class_losses = []
                    total_loss = []

                    # gpu N 개를 대비한 코드 (Data Parallelism)
                    for img, xcyc_target, wh_target, objectness, class_target, weights in zip(image_split, xcyc_split,
                                                                                              wh_split,
                                                                                              objectness_split,
                                                                                              class_split,
                                                                                              weights_split):
                        output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(
                            img)
                        xcyc_loss, wh_loss, object_loss, class_loss = loss(output1, output2, output3, xcyc_target,
                                                                           wh_target, objectness,
                                                                           class_target, weights)
                        xcyc_losses.append(xcyc_loss.asscalar())
                        wh_losses.append(wh_loss.asscalar())
                        object_losses.append(object_loss.asscalar())
                        class_losses.append(class_loss.asscalar())
                        total_loss.append(xcyc_loss + wh_loss + object_loss + class_loss)
                    if AMP:
                        with amp.scale_loss(total_loss, trainer) as scaled_loss:
                            autograd.backward(scaled_loss)
                    else:
                        autograd.backward(total_loss)

                    xcyc_all_losses.append(sum(xcyc_losses))
                    wh_all_losses.append(sum(wh_losses))
                    object_all_losses.append(sum(object_losses))
                    class_all_losses.append(sum(class_losses))

            trainer.step(batch_size=td_batch_size, ignore_stale_grad=False)
            # 비우기
            for p in net.collect_params().values():
                p.zero_grad()

            xcyc_loss_sum += sum(xcyc_all_losses) / td_batch_size
            wh_loss_sum += sum(wh_all_losses) / td_batch_size
            object_loss_sum += sum(object_all_losses) / td_batch_size
            class_loss_sum += sum(class_all_losses) / td_batch_size

            if batch_count % batch_log == 0:
                logging.info(f'[Epoch {i}][Batch {batch_count}/{train_update_number_per_epoch}],'
                             f'[Speed {td_batch_size / (time.time() - time_stamp):.3f} samples/sec],'
                             f'[Lr = {trainer.learning_rate}]'
                             f'[xcyc loss = {sum(xcyc_all_losses) / td_batch_size:.3f}]'
                             f'[wh loss = {sum(wh_all_losses) / td_batch_size:.3f}]'
                             f'[obj loss = {sum(object_all_losses) / td_batch_size:.3f}]'
                             f'[class loss = {sum(class_all_losses) / td_batch_size:.3f}]')
            time_stamp = time.time()

        train_xcyc_loss_mean = np.divide(xcyc_loss_sum, train_update_number_per_epoch)
        train_wh_loss_mean = np.divide(wh_loss_sum, train_update_number_per_epoch)
        train_object_loss_mean = np.divide(object_loss_sum, train_update_number_per_epoch)
        train_class_loss_mean = np.divide(class_loss_sum, train_update_number_per_epoch)
        train_total_loss_mean = train_xcyc_loss_mean + train_wh_loss_mean + train_object_loss_mean + train_class_loss_mean
        logging.info(
            f"train xcyc loss : {train_xcyc_loss_mean} / "
            f"train wh loss : {train_wh_loss_mean} / "
            f"train object loss : {train_object_loss_mean} / "
            f"train class loss : {train_class_loss_mean} / "
            f"train total loss : {train_total_loss_mean}"
        )

        if i % eval_period == 0 and valid_list:

            xcyc_loss_sum = 0
            wh_loss_sum = 0
            object_loss_sum = 0
            class_loss_sum = 0

            # loss 구하기
            for image, label, xcyc_all, wh_all, objectness_all, class_all, weights_all, _ in valid_dataloader:
                vd_batch_size, _, height, width = image.shape

                if GPU_COUNT <= 1:
                    image = gluon.utils.split_and_load(image, [ctx], even_split=False)
                    label = gluon.utils.split_and_load(label, [ctx], even_split=False)
                    xcyc_all = gluon.utils.split_and_load(xcyc_all, [ctx], even_split=False)
                    wh_all = gluon.utils.split_and_load(wh_all, [ctx], even_split=False)
                    objectness_all = gluon.utils.split_and_load(objectness_all, [ctx], even_split=False)
                    class_all = gluon.utils.split_and_load(class_all, [ctx], even_split=False)
                    weights_all = gluon.utils.split_and_load(weights_all, [ctx], even_split=False)
                else:
                    image = gluon.utils.split_and_load(image, ctx, even_split=False)
                    label = gluon.utils.split_and_load(label, ctx, even_split=False)
                    xcyc_all = gluon.utils.split_and_load(xcyc_all, ctx, even_split=False)
                    wh_all = gluon.utils.split_and_load(wh_all, ctx, even_split=False)
                    objectness_all = gluon.utils.split_and_load(objectness_all, ctx, even_split=False)
                    class_all = gluon.utils.split_and_load(class_all, ctx, even_split=False)
                    weights_all = gluon.utils.split_and_load(weights_all, ctx, even_split=False)

                xcyc_losses = []
                wh_losses = []
                object_losses = []
                class_losses = []
                total_loss = []

                # gpu N 개를 대비한 코드 (Data Parallelism)
                for img, lb, xcyc_target, wh_target, objectness, class_target, weights in zip(image, label, xcyc_all,
                                                                                              wh_all, objectness_all,
                                                                                              class_all, weights_all):
                    gt_box = lb[:, :, :4]
                    gt_id = lb[:, :, 4:5]

                    output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(
                        img)
                    id, score, bbox = prediction(output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2,
                                                 offset3, stride1, stride2, stride3)

                    precision_recall.update(pred_bboxes=bbox,
                                            pred_labels=id,
                                            pred_scores=score,
                                            gt_boxes=gt_box,
                                            gt_labels=gt_id)

                    xcyc_loss, wh_loss, object_loss, class_loss = loss(output1, output2, output3, xcyc_target,
                                                                       wh_target, objectness,
                                                                       class_target, weights)
                    xcyc_losses.append(xcyc_loss.asscalar())
                    wh_losses.append(wh_loss.asscalar())
                    object_losses.append(object_loss.asscalar())
                    class_losses.append(class_loss.asscalar())
                    total_loss.append(xcyc_losses + wh_losses + object_losses + class_losses)

                xcyc_loss_sum += sum(xcyc_losses) / vd_batch_size
                wh_loss_sum += sum(wh_losses) / vd_batch_size
                object_loss_sum += sum(object_losses) / vd_batch_size
                class_loss_sum += sum(class_losses) / vd_batch_size

            valid_xcyc_loss_mean = np.divide(xcyc_loss_sum, valid_update_number_per_epoch)
            valid_wh_loss_mean = np.divide(wh_loss_sum, valid_update_number_per_epoch)
            valid_object_loss_mean = np.divide(object_loss_sum, valid_update_number_per_epoch)
            valid_class_loss_mean = np.divide(class_loss_sum, valid_update_number_per_epoch)
            valid_total_loss_mean = valid_xcyc_loss_mean + valid_wh_loss_mean + valid_object_loss_mean + valid_class_loss_mean

            logging.info(
                f"valid xcyc loss : {valid_xcyc_loss_mean} / "
                f"valid wh loss : {valid_wh_loss_mean} / "
                f"valid object loss : {valid_object_loss_mean} / "
                f"valid class loss : {valid_class_loss_mean} / "
                f"valid total loss : {valid_total_loss_mean}"
            )

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
                image, label, _, _, _, _, _, _ = next(dataloader_iter)
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
                for img, lb in zip(image, label):
                    gt_boxes = lb[:, :, :4]
                    gt_ids = lb[:, :, 4:5]
                    output1, output2, output3, anchor1, anchor2, anchor3, offset1, offset2, offset3, stride1, stride2, stride3 = net(
                        img)
                    ids, scores, bboxes = prediction(output1, output2, output3, anchor1, anchor2, anchor3, offset1,
                                                     offset2, offset3, stride1, stride2, stride3)

                    for ig, gt_id, gt_box, id, score, bbox in zip(img, gt_ids, gt_boxes, ids, scores, bboxes):
                        ig = ig.transpose(
                            (1, 2, 0)) * mx.nd.array(std, ctx=ig.context) + mx.nd.array(mean, ctx=ig.context)
                        ig = (ig * 255).clip(0, 255)

                        # ground truth box 그리기
                        ground_truth = plot_bbox(ig, gt_box, scores=None, labels=gt_id, thresh=None,
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

                summary.add_image(tag="valid_result", image=np.array(batch_image), global_step=i)

                summary.add_scalar(tag="xy_loss", value={"train_xcyc_loss": train_xcyc_loss_mean,
                                                         "valid_xcyc_loss": valid_xcyc_loss_mean}, global_step=i)
                summary.add_scalar(tag="wh_loss", value={"train_wh_loss": train_wh_loss_mean,
                                                         "valid_wh_loss": valid_wh_loss_mean}, global_step=i)
                summary.add_scalar(tag="object_loss", value={"train_object_loss": train_object_loss_mean,
                                                             "valid_object_loss": valid_object_loss_mean},
                                   global_step=i)
                summary.add_scalar(tag="class_loss", value={"train_class_loss": train_class_loss_mean,
                                                            "valid_class_loss": valid_class_loss_mean}, global_step=i)

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
            '''
                mxnet1.6.0 버전 에서 AMP 사용시 위에 미리 선언한 prediction을 사용하면 문제가 될 수 있다. 
                -yolo v3, gaussian yolo v3 에서는 문제가 발생한다.
                mxnet 1.5.x 버전에서는 아래와 같이 새로 선언하지 않아도 정상 동작한다.  

                block들은 함수 인자로 보낼 경우 자기 자신이 보내진다.(복사되는 것이 아님)
                export_block_for_cplusplus 에서 prediction 이 hybridize 되면서 
                미리 선언한 prediction도 hybridize화 되면서 symbol 형태가 된다. 
                이런 현상을 보면 아래와같이 다시 선언해 주는게 맞는 것 같다.
            '''
            auxnet = Prediction(
                from_sigmoid=False,
                num_classes=num_classes,
                nms_thresh=nms_thresh,
                nms_topk=nms_topk,
                except_class_thresh=except_class_thresh,
                multiperclass=multiperclass)
            postnet = PostNet(net=net, auxnet=auxnet)

            try:
                net.export(os.path.join(weight_path, f"{model}"), epoch=i, remove_amp_cast=True)  # for onnx
                net.save_parameters(os.path.join(weight_path, f"{i}.params"))  # onnx 추출용
                # network inference, decoder, nms까지 처리됨 - mxnet c++에서 편리함 / onnx로는 추출 못함.
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
        offset_alloc_size=(64, 64),
        anchors={"shallow": [(10, 13), (16, 30), (33, 23)],
                 "middle": [(30, 61), (62, 45), (59, 119)],
                 "deep": [(116, 90), (156, 198), (373, 326)]},
        graphviz=False,
        epoch=10,
        input_size=[416, 416],
        batch_log=100,
        batch_size=16,
        batch_interval=10,
        subdivision=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        multiscale=False,
        factor_scale=[13, 5],
        ignore_threshold=0.5,
        dynamic=False,
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        GPU_COUNT=0,
        Darknetlayer=53,
        pretrained_base=True,
        pretrained_path="modelparam",
        AMP=True,
        valid_size=8,
        eval_period=5,
        tensorboard=True,
        valid_graph_path="valid_Graph",
        valid_html_auto_open=True,
        using_mlflow=True,
        multiperclass=True,
        nms_thresh=0.5,
        nms_topk=500,
        iou_thresh=0.5,
        except_class_thresh=0.05,
        plot_class_thresh=0.5)
