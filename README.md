>## ***One Stage Detector***
* [***Single Shot MultiBox Detector***](https://github.com/DeepFocuser/Mxnet-Detector/tree/master/SSD)
* [***YoloV3 Detector***](https://github.com/DeepFocuser/Mxnet-Detector/tree/master/YoloV3)
* [***Gaussian YoloV3 Detector***](https://github.com/DeepFocuser/Mxnet-Detector/tree/master/GaussianYoloV3)
* [***Retina Detector***](https://github.com/DeepFocuser/Mxnet-Detector/tree/master/RETINA)
* [***Efficient Detector***](https://github.com/DeepFocuser/Mxnet-Detector/tree/master/Efficient)
* [***Center Detector***](https://github.com/DeepFocuser/Mxnet-Detector/tree/master/Center)

>## ***Development environment***
* OS : ubuntu linux 16.04 LTS
* Graphic card / driver : rtx 2080ti / 418.56
* Anaconda version : 4.7.12
    * Configure Run Environment
        1. Create a virtual environment
        ```cmd
        jg@JG:~$ conda create -n mxnet python==3.7.3
        ```
        2. Install the required module
        ```cmd
        jg@JG:~$ conda activate mxnet
        (mxnet) jg@JG:~$ conda install cudatoolkit==10.1.243 cudnn 
        (mxnet) jg@JG:~$ pip install mxnet-cu101==1.6.0 tensorboard mxboard gluoncv plotly mlflow opencv-python==4.1.1.26 onnx tqdm PyYAML --pre --upgrade
        ```

>## ***Author*** 

* medical18@naver.com / JONGGON
