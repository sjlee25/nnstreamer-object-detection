## Object Detection with NNStreamer

### 1. Introduction

Object detection demo using NNStreamer and GStreamer framework.

<br>

### 2. Requirements

**GStreamer** and **NNStreamer** should be installed to run this demo.

<br>

For ubuntu users, install nnstreamer with PPA repository as:

```bash
$ sudo apt-add-repository ppa:nnstreamer
$ sudo apt install nnstreamer nnstreamer-tensorflow
```

All available nnstreamer plugins are listed in [official install guidelines for PPA repository](https://github.com/nnstreamer/nnstreamer/blob/main/Documentation/getting-started-ubuntu-ppa.md).

<br>

To use your own tensorflow libraries instead of ones given by nnstreamer-tensorflow package,

you need to build nnstreamer from source. You can refer to [install guidelines with Meson/Ninja build](https://github.com/nnstreamer/nnstreamer/blob/main/Documentation/getting-started-meson-build.md).

<br>

### 3. Usage

1. Clone this repository.

```bash
$ git clone https://github.com/lsj1213m/NNStreamer.git
```

<br>

2. Install required packages with pip.

```bash
$ pip install -r ./requirements.txt
```

<br>

3. Run ```object_detection.py``` with options you want to demo video detection.

```bash
$ python object_detection.py --video [path to video file] [other options]
```

​	For example, you can run as follows if you want to use 0-th GPU with YOLOv3 model.

```bash
$ python object_detection.py --video ./video/test_video_street.mp4 --model yolo --device gpu --gpu_idx 0
```

<br>

​	All available options with descriptions are here.

```bash
--video   [path to video file]: input video file path
--use_webcam: whether to use webcam or not (default: False)
--model   ['ssdlite'/'yolo_tiny'/'yolo']: model name to use
--score   [threshold value]: threshold for score (default: 0.3)
--device  ['cpu'/'gpu']: device to use for inference (default: 'cpu')
--gpu_idx ['0'/'1'/...]: gpu device number to use if the gpu will be used (default: '0')
```

​	These options and other model-specific settings can be modified in ```config.py```.

<br>

### 4. Result



<br>

### 5. TODO
- FPS, mAP measurements
- Apply more  object detection models
- Collaborate with TVM framework

<br>

---

### Credits

#### Github Repositories

- [nnstreamer/nnstreamer (NNStreamer Official Github)](https://github.com/nnstreamer/nnstreamer)

- [trekhleb/machine-learning-experiments](https://github.com/trekhleb/machine-learning-experiments)

- [YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)

- [wizyoung/YOLOv3_TensorFlow](https://github.com/wizyoung/YOLOv3_TensorFlow)
- [zzh8829/yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)

<br>

#### Papers

- [NNStreamer: Stream Processing Paradigm for Neural Networks](https://arxiv.org/abs/1901.04985)

- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

<br>

#### Frameworks

- [NNStreamer](https://nnstreamer.ai/) & [GStreamer](https://gstreamer.freedesktop.org/)

- [OpenCV](https://opencv.org/)

- [Tensorflow](https://www.tensorflow.org/)

<br>

#### Datasets

- [ILSVRC 2015](http://image-net.org/challenges/LSVRC/2015/)

- [COCO 2017](https://cocodataset.org/)
