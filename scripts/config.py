#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random 
import numbers

import torchvision.transforms as transforms

directory_root = os.path.dirname(os.path.realpath(__file__)) + "/../"
model_path = directory_root + "/epochs/"
image_path = directory_root + "/test_images/"
result_path = directory_root + "/results/"

from PIL import Image

class SquareZeroPadding(object):

    def __init__(self, fill = 0):

        assert isinstance(fill, (numbers.Number, str, tuple))
        self.fill = fill

    def __call__(self, img):

        """
        Args:
            img (PIL Image): Image to be padded.
        Returns:
            PIL Image: Zero Padded Squared image.
        """

        fill_color = (self.fill, self.fill, self.fill)
        x, y = img.size
        _size = max(x, y)
        padded_img = Image.new('RGB', (_size, _size), fill_color)
        padded_img.paste(img, ((_size - x) / 2, (_size - y) / 2))
        return padded_img

Normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )  

_preprocess = transforms.Compose([
    SquareZeroPadding(),
    transforms.Resize((224, 224), 2),
    transforms.ToTensor(),
    Normalize
])

class Config():

    preprocess = _preprocess
    model_dir = model_path
    image_dir = image_path
    result_dir = result_path
    cnn_model = "/epoch_2_100.pth"
    upscale_factor = 2
