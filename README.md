# <img src="assets/logo.png" width=50> Cam2BEV

<img src="assets/teaser.gif" align="right" width=320 height=200>

This repository contains the official implementation of our methodology for the computation of a semantically segmented bird's eye view (BEV) image given the images of multiple vehicle-mounted cameras as presented in our paper:

> **A Sim2Real Deep Learning Approach for the Transformation of Images from Multiple Vehicle-Mounted Cameras to a Semantically Segmented Image in Bird’s Eye View**
([IEEE Xplore](https://ieeexplore.ieee.org/document/9294462), [arXiv](https://arxiv.org/abs/2005.04078))  
>
> [Lennart Reiher](https://github.com/lreiher), [Bastian Lampe](https://www.ika.rwth-aachen.de/en/institute/staff/bastian-lampe-m-sc.html), and [Lutz Eckstein](https://www.ika.rwth-aachen.de/en/institute/management/univ-prof-dr-ing-lutz-eckstein.html)  
> [Institute for Automotive Engineering (ika), RWTH Aachen University](https://www.ika.rwth-aachen.de/en/)

[![Cam2BEV_video](assets/video_preview.jpg)](https://youtu.be/TzXuwt56a0E)

> _**Abstract**_ — Accurate environment perception is essential for automated driving. When using monocular cameras, the distance estimation of elements in the environment poses a major challenge. Distances can be more easily estimated when the camera perspective is transformed to a bird's eye view (BEV). For flat surfaces, _Inverse Perspective Mapping_ (IPM) can accurately transform images to a BEV. Three-dimensional objects such as vehicles and vulnerable road users are distorted by this transformation making it difficult to estimate their position relative to the sensor. This paper describes a methodology to obtain a corrected 360° BEV image given images from multiple vehicle-mounted cameras. The corrected BEV image is segmented into semantic classes and includes a prediction of occluded areas. The neural network approach does not rely on manually labeled data, but is trained on a synthetic dataset in such a way that it generalizes well to real-world data. By using semantically segmented images as input, we reduce the reality gap between simulated and real-world data and are able to show that our method can be successfully applied in the real world. Extensive experiments conducted on the synthetic data demonstrate the superiority of our approach compared to IPM.

We hope our paper, data and code can help in your research. If this is the case, please cite:
```
@INPROCEEDINGS{ReiherLampe2020Cam2BEV,
  author={L. {Reiher} and B. {Lampe} and L. {Eckstein}},
  booktitle={2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)}, 
  title={A Sim2Real Deep Learning Approach for the Transformation of Images from Multiple Vehicle-Mounted Cameras to a Semantically Segmented Image in Bird’s Eye View}, 
  year={2020},
  doi={10.1109/ITSC45102.2020.9294462}}
```
## Content

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Neural Network Architectures](#neural-network-architectures)
- [Customization](#customization)

## Repository Structure

```
Cam2BEV
├── data                        # where our synthetic datasets are downloaded to by default  
├── model                       # training scripts and configurations
│   ├── architecture                # TensorFlow implementations of neural network architectures
│   └── one_hot_conversion          # files defining the one-hot encoding of semantically segmented images
└── preprocessing               # preprocessing scripts
    ├── camera_configs              # files defining the intrinsics/extrinsics of the cameras used in our datasets
    ├── homography_converter        # script to convert an OpenCV homography for usage within the uNetXST SpatialTransformers
    ├── ipm                         # script for generating a classical homography image by means of IPM
    └── occlusion                   # script for introducing an occluded class to the BEV images
```

## Installation

We suggest to setup a **Python 3.8** virtual environment (e.g. by using _virtualenv_ or _conda_). Inside the virtual environment, users can then use _pip_ to install all package dependencies. The most important packages are _TensorFlow 2.7_ and _OpenCV 4.5_
```bash
pip install -r requirements.txt
```

## Data

We provide two synthetic datasets, which can be used to train the neural networks. The datasets are hosted in the [Cam2BEV Data Repository](https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data). Both datasets were used to produce the results presented in our paper:
- [*Dataset 1_FRLR*](https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data/-/tree/master/1_FRLR): images from four vehicle-mounted cameras, ground-truth BEV image centered above the ego vehicle
- [*Dataset 2_F*](https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data/-/tree/master/2_F): images from one frontal vehicle camera; ground-truth BEV image left-aligned with ego vehicle

For more information regarding the data, please refer to the [repository's README](https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data).

Both datasets can easily be downloaded and extracted by running the provided download script:
```bash
./data/download.sh
```

_**Note**: Download size is approximately 3.7GB, uncompressed size of both datasets is approximately 7.7GB._

## Preprocessing

Our paper describes two preprocessing techniques:  
(1) introducing an _occluded_ class to the label images and  
(2) generating the homography image.

### 1) Dealing with Occlusions

Traffic participants and static obstacles may occlude parts of the environment making predictions for those areas in a BEV image mostly impossible. In order to formulate a well-posed problem, an additional semantic class needs to be introduced to the label images for areas in BEV, which are occluded in the camera perspectives. To this end, [preprocessing/occlusion](preprocessing/occlusion/) can be used. See below for an example of the occlusion preprocessing.

![original](preprocessing/occlusion/assets/example-original.png) ![occluded](preprocessing/occlusion/assets/example-occluded.png)


Run the following command to process the original label images of _dataset 1_FRLR_ and introduce an _occluded_ class. You need to provide camera intrinsics/extrinsics for the drone camera and all vehicle-attached cameras (in the form of the yaml files).

_**Note**: In batch mode, this script utilizes multiprocessing. It can however still take quite some time to process the entire dataset. Therefore, we also provide already preprocessed data._

```bash
cd preprocessing/occlusion
```
```bash
./occlusion.py \
    --batch ../../data/1_FRLR/train/bev \
    --output ../../data/1_FRLR/train/bev+occlusion \
    ../camera_configs/1_FRLR/drone.yaml \
    ../camera_configs/1_FRLR/front.yaml \
    ../camera_configs/1_FRLR/rear.yaml \
    ../camera_configs/1_FRLR/left.yaml \
    ../camera_configs/1_FRLR/right.yaml
```

See [preprocessing/occlusion/README.md](preprocessing/occlusion/README.md) for more information.

### 2) Projective Preprocessing

As part of the incorporation of the Inverse Perspective Mapping (IPM) technique into our methods, the homographies, i.e. the projective transformations between vehicle camera frames and BEV need to be computed. As a preprocessing step to the first variation of our approach (Section III-C), IPM is applied to all images from the vehicle cameras. The transformation is set up to capture the same field of view as the ground truth BEV image. To this end, [preprocessing/ipm](preprocessing/ipm) can be used. See below for an example homography image computed from images of four vehicle-mounted cameras.
___
作为将逆视角映射（IPM）技术纳入我们的方法的一部分，homographies需要进行计算，即车载摄像头框架和BEV之间的投影变换。作为我们方法的第一个变体（第III-C节）的预处理步骤，IPM应用于来自车载摄像头的所有图像。设置变换以捕获与地面实况 BEV 图像相同的视场。为此，可以使用预处理/ipm。有关从四个车载摄像头的图像中计算出的示例同调图像，请参见下文。

![ipm](preprocessing/ipm/assets/example.png)

Run the following command to compute a homography BEV image from all camera images of _dataset 1_FRLR_. You need to provide camera intrinsics/extrinsics for the drone camera and all vehicle-attached cameras (in the form of the yaml files).

_**Note**: To save time, we also provide already preprocessed data._

```bash
cd preprocessing/ipm
```
```bash
./ipm.py --batch --cc \
    --output ../../data/1_FRLR/train/homography \
    --drone ../camera_configs/1_FRLR/drone.yaml \
    ../camera_configs/1_FRLR/front.yaml \
    ../../data/1_FRLR/train/front \
    ../camera_configs/1_FRLR/rear.yaml \
    ../../data/1_FRLR/train/rear \
    ../camera_configs/1_FRLR/left.yaml \
    ../../data/1_FRLR/train/left \
    ../camera_configs/1_FRLR/right.yaml \
    ../../data/1_FRLR/train/right
```

See [preprocessing/ipm/README.md](preprocessing/ipm/README.md) for more information.

## Training

Use the scripts [model/train.py](model/train.py), [model/evaluate.py](model/evaluate.py), and [model/predict.py](model/predict.py) to train a model, evaluate it on validation data, and make predictions on a testing dataset.

Input directories, training parameters, and more can be set via CLI arguments or in a config file. Run the scripts with `--help`-flag or see one of the provided exemplary config files for reference. We provide config files for either one of the networks and datasets:
- [model/config.1_FRLR.deeplab-mobilenet.yml](model/config.1_FRLR.deeplab-mobilenet.yml)
- [model/config.1_FRLR.deeplab-xception.yml](model/config.1_FRLR.deeplab-mobilenet.yml)
- [model/config.1_FRLR.unetxst.yml](model/config.1_FRLR.unetxst.yml)
- [model/config.2_F.deeplab-mobilenet.yml](model/config.2_F.deeplab-mobilenet.yml)
- [model/config.2_F.deeplab-xception.yml](model/config.2_F.deeplab-xception.yml)
- [model/config.2_F.unetxst.yml](model/config.2_F.unetxst.yml)

The following commands will guide you through training _uNetXST_ on _dataset 1_FRLR_.

### Training

Start training _uNetXST_ by passing the provided config file [model/config.1_FRLR.unetxst.yml](model/config.1_FRLR.unetxst.yml). Training will automatically stop if the MIoU score on the validation dataset is not rising anymore.

```bash
cd model/
```
```bash
./train.py -c config.1_FRLR.unetxst.yml
```

You can visualize training progress by pointing *TensorBoard* to the output directory (`model/output` by default). Training metrics will also be printed to `stdout`.

### Evaluation

Before evaluating your trained model, set the parameter `model-weights` to point to the `best_weights.hdf5` file in the `Checkpoints` folder of its model directory. Then run evaluation to compute a confusion matrix and class IoU scores.

```bash
./evaluate.py -c config.1_FRLR.unetxst.yml --model-weights output/<YOUR-TIMESTAMP>/Checkpoints/best_weights.hdf5
```

The evaluation results will be printed at the end of evaluation and also be exported to the `Evaluation` folder in your model directory.

### Testing

To actually see the predictions your network makes, try it out on unseen input images, such as the validation dataset. The predicted BEV images are exported to the directory specified by the parameter `output-dir-testing`.

```bash
./predict.py -c config.1_FRLR.unetxst.yml --model-weights output/<YOUR-TIMESTAMP>/Checkpoints/best_weights.hdf5 --prediction-dir output/<YOUR-TIMESTAMP>/Predictions
```

## Neural Network Architectures

We provide implementations for the use of the neural network architectures _DeepLab_ and _uNetXST_ in [model/architecture](model/architecture). _DeepLab_ comes with two different backbone networks: _MobileNetV2_ or _Xception_.

### DeepLab

The _DeepLab_ models are supposed to take the homography images computed by Inverse Perspective Mapping ([preprocessing/ipm](preprocessing/ipm)) as input.

#### Configuration
- set `model` to `architecture/deeplab_mobilenet.py` or `architecture/deeplab_xception.py`
- set `input-training` and the other input directory parameters to the folders containing the homography images
- comment `unetxst-homographies` in the config file or don't supply it via CLI, respectively 

### uNetXST

The _uNetXST_ model contains SpatialTransformer units, which perform IPM inside the network. Therefore, when building the network, the homographies to transform images from each camera need to be provided.
___
_uNetXST_ 模型包含 SpatialTransformer 单元，这些单元在网络内部执行 IPM。因此，在构建网络时，需要提供从每个摄像机转换图像的同形异义词。

#### Configuration
- set `model` to `architecture/uNetXST.py`
- set `input-training` and the other input directory parameters to a list of folders containing the images from each camera (e.g. `[data/front, data/rear, data/left, data/right]`)
- set `unetxst-homographies` to a Python file containing the homographies as a list of NumPy arrays stored in a variable `H` (e.g. `../preprocessing/homography_converter/uNetXST_homographies/1_FRLR.py`)  
  - we provide these homographies for our two datasets in [preprocessing/homography_converter/uNetXST_homographies/1_FRLR.py](preprocessing/homography_converter/uNetXST_homographies/1_FRLR.py) and [preprocessing/homography_converter/uNetXST_homographies/2_F.py](preprocessing/homography_converter/uNetXST_homographies/2_F.py)
  - in order to compute these homographies for different camera configurations, follow the instructions in [preprocessing/homography_converter](preprocessing/homography_converter)

## Customization

#### _I want to set different training hyperparameters_

Run the training script with `--help`-flag or have a look at one of the provided exemplary config files to see what parameters you can easily set.

#### _I want the networks to work on more/fewer semantic classes_

The image datasets we provide include all 30 _CityScapes_ class colors. How these are reduced to say 10 classes is defined in the one-hot conversion files in [model/one_hot_conversion](model/one_hot_conversion). Use the training parameters `--one-hot-palette-input` and `--one-hot-palette-label` to choose one of the files. You can easily create your own one-hot conversion file, they are quite self-explanatory.

If you adjust `--one-hot-palette-label`, you will also need to modify `--loss-weights`. Either omit the parameter to weight all output classes evenly, or compute new suitable loss weights. The weights found in the provided config files were computed (from the `model` directory) with the following Python snippet.
```python
import numpy as np
import utils
palette = utils.parse_convert_xml("one_hot_conversion/convert_9+occl.xml")
dist = utils.get_class_distribution("../data/1_FRLR/train/bev+occlusion", (256, 512), palette)
weights = np.log(np.reciprocal(list(dist.values())))
print(weights)
```

#### _I want to use my own data_

You will need to run the preprocessing methods on your own data. A rough outline on what you need to consider:
- specify camera intrinsics/extrinsics similar to the files found in [preprocessing/camera_configs]([preprocessing/camera_configs])
- run [preprocessing/occlusion/occlusion.py](preprocessing/occlusion/occlusion.py)
- run [preprocessing/occlusion/ipm.py](preprocessing/occlusion/ipm.py)
- compute uNetXST-compatible homographies by following the instructions in [preprocessing/homography_converter](preprocessing/homography_converter)
- adjust or create a new one-hot conversion file ([model/one_hot_conversion](model/one_hot_conversion))
- set all training parameters in a dedicated config file
- start training


# 实验过程
- 利用ipm.py文件计算opencv homography，得到如下结果（此过程缺少俯视摄像头参数）
```
./ipm.py -v ../camera_configs/0_HW/left_back.yml rear ../camera_configs/0_HW/left_front.yml left ../camera_configs/0_HW/right_back.yml right ../camera_configs/0_HW/right_front.yml front

OpenCV homography for rear:
[[0.4174712299002103, -0.09310665839863179, -18.22150823766197], [0.18200114807706824, -0.06174939185829669, -17.99062602824388], [0.0009734955405027581, -0.00018581844620112566, -0.10740308656633131]]
OpenCV homography for left:
[[0.4960821291978444, -0.00704347330818303, -480.7343628059732], [0.21171638392114545, -0.03452077250374634, -187.6388898927711], [0.0011052851994656054, -5.814868921805794e-05, -1.0964208682637886]]
OpenCV homography for right:
[[-0.421055551338769, -0.21569638804448146, 515.1102112188419], [-0.19716858346068017, -0.08593641371065266, 256.5431550689641], [-0.0009742106553017512, -0.0004237089184009168, 1.1519641220836545]]
OpenCV homography for front:
[[-0.5094844647701582, -0.1474041542017286, 203.4304209338446], [-0.23397980786942155, -0.07984661366564264, 73.59441187657308], [-0.0011338714752396123, -0.00039917305171821636, 0.4667389678019258]]
```

- 利用homography_converter将opencv的参数转换为stn网络使用的参数：
  - rear
  ```
  Adjusted SpatialTransformer homography usable for resolution 256x512 -> 256x512:
  [[-19.213813215249033, 1.240059165734868, -4.0638204496654815], [-21.094477687887352, -31.78323293285227, -24.14709503337365], [23.881569390170178, -5.933275123892921, 4.671038000324768]]
  ```
  - left
  ```
  [[13.589055062501462, 4.412136956178291, 7.79540236626548], [40.33543416981511, -26.331688686232017, -1.8127470569043034], [21.616398475941114, 8.808013149564966, 11.093595960595376]]
  ```
  - right
  ```
  [[-0.421055551338769, -0.21569638804448146, 515.1102112188419], [-0.19716858346068017, -0.08593641371065266, 256.5431550689641], [-0.0009742106553017512, -0.0004237089184009168, 1.1519641220836545]]
  ```
  - front
  ```
  [[-12.074394738320724, 6.697303613989412, -1.759563216838852], [61.424022049579776, 7.342386035309095, 21.824772957105104], [2.2752247770098735, -14.399838923380008, -6.236124136333338]]
  ```