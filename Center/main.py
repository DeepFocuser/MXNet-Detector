import os

import mlflow as ml
import mxnet as mx
import yaml

import test
import train

# MXNET-ONNX EXPORT 지원 가능 함수 확인
# -> https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/contrib/onnx/mx2onnx

'''
MXNET_CUDNN_AUTOTUNE_DEFAULT
Values: 0, 1, or 2 (default=1)
The default value of cudnn auto tuning for convolution layers.
Value of 0 means there is no auto tuning to pick the convolution algorithm
Performance tests are run to pick the convolution algo when value is 1 or 2
Value of 1 chooses the best algo in a limited workspace
Value of 2 chooses the fastest algo whose memory requirements may be larger than the default workspace threshold
'''
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "2"  # 메모리 사용 관련
stream = yaml.load(open("configs/detector.yaml", "rt", encoding='UTF8'), Loader=yaml.SafeLoader)

# dataset
parser = stream['Dataset']
train_dataset_path = parser['train']
valid_dataset_path = parser['valid']
test_dataset_path = parser['test']
test_weight_path = parser['test_weight_path']

test_save_path = parser['save_path']
save_flag = parser['save_flag']
show_flag = parser['show_flag']
topk = parser['topk']
plot_class_thresh = parser['plot_class_thresh']
test_graph_path = parser["test_graph_path"]

# model
parser = stream['model']
training = parser["training"]
load_name = parser["load_name"]
save_period = parser["save_period"]
load_period = parser["load_period"]
input_size = parser["input_size"]
base = parser["ResNetbase"]
pretrained_base = parser["pretrained_base"]
pretrained_path = parser["pretrained_path"]
graphviz = parser["graphviz"]

# hyperparameters
parser = stream['hyperparameters']

image_mean = parser["image_mean"]
image_std = parser["image_std"]

epoch = parser["epoch"]
batch_size = parser["batch_size"]
batch_log = parser["batch_log"]
batch_interval = parser["batch_interval"]
subdivision = parser["subdivision"]
multiscale = parser["multiscale"]
factor_scale = parser["factor_scale"]
data_augmentation = parser["data_augmentation"]
num_workers = parser["num_workers"]
optimizer = parser["optimizer"]
lambda_off = parser["lambda_off"]
lambda_size = parser["lambda_size"]
learning_rate = parser["learning_rate"]
decay_lr = parser["decay_lr"]
decay_step = parser["decay_step"]
AMP = parser["AMP"]

# gpu vs cpu
parser = stream['context']
using_cuda = parser["using_cuda"]

parser = stream['validation']
valid_size = parser["valid_size"]
eval_period = parser["eval_period"]
tensorboard = parser["tensorboard"]
valid_graph_path = parser["valid_graph_path"]

parser = stream['mlflow']
using_mlflow = parser["using_mlflow"]
run_name = parser["run_name"]

if mx.context.num_gpus() > 0 and using_cuda:
    GPU_COUNT = mx.context.num_gpus()
else:
    GPU_COUNT = 0

# window 운영체제에서 freeze support 안나오게 하려면, 아래와 같이 __name__ == "__main__" 에 해줘야함.
if __name__ == "__main__":

    print("\n실행 경로 : " + __file__)
    if training:
        if using_mlflow:
            ml.set_tracking_uri("./mlruns")  # mlruns가 기본 트래킹이다.
            ex_id = ml.set_experiment("CENTER_" + "RES" + str(base))
            ml.start_run(run_name=run_name, experiment_id=ex_id)
            ml.log_param("height", input_size[0])
            ml.log_param("width", input_size[1])
            ml.log_param("pretrained_base", pretrained_base)
            ml.log_param("train dataset path", train_dataset_path)
            ml.log_param("valid dataset path", valid_dataset_path)
            ml.log_param("test dataset path", test_dataset_path)
            ml.log_param("epoch", epoch)
            ml.log_param("batch size", batch_size)
            ml.log_param("multiscale", multiscale)
            ml.log_param("data augmentation", data_augmentation)
            ml.log_param("optimizer", optimizer)
            ml.log_param("learning rate", learning_rate)
            ml.log_param("decay lr", decay_lr)

        train.run(mean=image_mean,
                  std=image_std,
                  graphviz=graphviz,
                  epoch=epoch,
                  input_size=input_size,
                  batch_size=batch_size,
                  batch_log=batch_log,
                  batch_interval=batch_interval,
                  subdivision=subdivision,
                  train_dataset_path=train_dataset_path,
                  valid_dataset_path=valid_dataset_path,
                  multiscale=multiscale,
                  factor_scale=factor_scale,
                  data_augmentation=data_augmentation,
                  num_workers=num_workers,
                  optimizer=optimizer,
                  lambda_off=lambda_off,
                  lambda_size = lambda_size,
                  save_period=save_period,
                  load_period=load_period,
                  learning_rate=learning_rate,
                  decay_lr=decay_lr,
                  decay_step=decay_step,
                  GPU_COUNT=GPU_COUNT,
                  base=base,
                  pretrained_base=pretrained_base,
                  pretrained_path=pretrained_path,
                  AMP=AMP,

                  valid_size=valid_size,
                  eval_period=eval_period,
                  tensorboard=tensorboard,
                  valid_graph_path=valid_graph_path,
                  using_mlflow=using_mlflow,

                  # valid dataset 그리기
                  topk=topk,
                  plot_class_thresh=plot_class_thresh)

        if using_mlflow:
            ml.end_run()
    else:
        test.run(mean=image_mean,
                 std=image_std,
                 load_name=load_name, load_period=load_period, GPU_COUNT=GPU_COUNT,
                 test_weight_path=test_weight_path,
                 test_dataset_path=test_dataset_path, num_workers=num_workers,
                 test_save_path=test_save_path,
                 test_graph_path=test_graph_path,
                 lambda_off=lambda_off,
                 lambda_size=lambda_size,
                 show_flag=show_flag,
                 save_flag=save_flag,
                 # test dataset 그리기
                 topk=topk,
                 plot_class_thresh=plot_class_thresh)
