# Tensorrt-robotic-grasping
This example deploys Tensorrt into the code of Antipodal Robotic Grasp：
## Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network

Sulabh Kumra, Shirin Joshi, Ferat Sahin

[arxiv](https://arxiv.org/abs/1909.04810) | [video](https://youtu.be/cwlEhdoxY4U)

## Introduction
- The 01 folder contains some pictures of the related datasets.
- Onnxtotrt.py is used to convert grcnn's model files to ONNX.
- Onnx-tensorrt needs to be downloaded to convert onnx to TRT files.
- This example only uses rgb images to train the network, you can use rgbd images to train the network, but you need to modify the inferencetest.py and predict.py related content.In order for users to modified by themselves, the onnx-tensorrt and tensorrt used in this environment are also uploaded to the author's GitHub repositories.

## Tested environment
Ubuntu18.04+cuda11.3+cudnn8.6.0+python3.8+torch1.12.0+TensorRT8.5.2.2

## Installation
git clone
```bash
$ git clone https://github.com/XiongCaiyu/Tensorrt-robotic-grasping.git
```
create environment
```bash
$ conda create -n pt12 python=3.8
```
activate enviroment
```bash
$ conda activate pt12
```
install Pytorch

[Pytorch=1.12.0](https://pytorch.org/)

### (optional)Additional development
Download tensorrt from BaiDuDisk
```bash
$ https://pan.baidu.com/s/1Ln8BNA95-FMZjFUZ3qXphQ?pwd=ltjy 
$ password = ltjy
```
Uzip and remember to add PATH and LD_LIBRARY_PATH to your bashrc

Download onnx-tensorrt kit
```bash
$ git clone https://github.com/onnx/onnx-tensorrt.git
$ cd onnx-tensorrt
$ git checkout 8.0-GA
$ git submodule update --init
$ mkdir build && cd build
$ cmake .. -DTENSORRT_ROOT=<your tensorrt PATH>
$ make -j8
$ sudo make install
```

##  Requirement
- numpy
- opencv-python
- matplotlib
- scikit-image
- imageio
- torch
- torchvision
- torchsummary
- tensorboardX
- pyrealsense2
- Pillow
- Tensorrt

## Run
```bash
$ python predict.py
```


