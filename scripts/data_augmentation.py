#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numbers
from PIL import Image
import torchvision.transforms as transforms

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

RandomColorJitter = transforms.Lambda(

    lambda x: transforms.ColorJitter(brightness = 0.1, contrast = 0.1, hue = 0.01)(x) if random.random() < 0.5 else x)

RandomZoom = transforms.Lambda(

    lambda x: transforms.Resize((224, 224), 2)(transforms.CenterCrop((220, 220))(x)) if random.random() < 0.5 else x)
