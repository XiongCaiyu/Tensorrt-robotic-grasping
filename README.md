# Tensorrt-robotiic-grasping
This example deploys Tensorrt into the code of Antipodal Robotic Grasp：
## Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network

Sulabh Kumra, Shirin Joshi, Ferat Sahin

[arxiv](https://arxiv.org/abs/1909.04810) | [video](https://youtu.be/cwlEhdoxY4U)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/antipodal-robotic-grasping-using-generative/robotic-grasping-on-cornell-grasp-dataset)]

## Introduction
- The 01 folder contains some pictures of the related datasets
- Onnxtotrt.py is used to convert grcnn's model files to ONNX
- Onnx-tensorrt needs to be downloaded to convert onnx to TRT files

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

## Tested environment
Ubuntu18.04+cuda11.3+cudnn8.6.0

## Run
```bash
$ python predict.py
```

This example only uses rgb images to train the network, you can use rgbd images to train the network, but you need to modify the inferencetest.py and predict.py related content.In order for users to modified by themselves, the onnx-tensorrt and tensorrt used in this environment are also uploaded to the author's GitHub repository.

