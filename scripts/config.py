#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random 
import torchvision.transforms as transforms

from data_augmentation import SquareZeroPadding, Normalize, RandomColorJitter, RandomZoom

directory_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
model_path = directory_root + "/epochs/"
image_path = directory_root + "/test_images/"
result_path = directory_root + "/results/"

_size_preprocess = transforms.Compose([
    SquareZeroPadding(),
    transforms.Resize((224, 224), 2),
])

_to_tensor_preprocess = transforms.Compose([
    transforms.ToTensor(),
    Normalize
])

_random_data_aug_preprocess = transforms.Compose([
    RandomZoom,
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees = 10),
    RandomColorJitter,    
])

class Config():

    size_preprocess = _size_preprocess
    to_tensor_preprocess = _to_tensor_preprocess
    random_data_aug_preprocess = _random_data_aug_preprocess
    model_dir = model_path
    image_dir = image_path
    result_dir = result_path
    upscale_factor = 3
    cnn_model = "/epoch_" + str(upscale_factor) + "_100.pth"
