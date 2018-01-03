#!/usr/bin/env python
# -*- coding: utf-8 -*-

from config import Config
from espcn_model import Net
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor

import numpy as np

def espcn_img(img_path, espcn_model):

    img = Image.open(img_path).convert('YCbCr')
    y, cb, cr = img.split()
    img = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    img = img.cuda()

    out = espcn_model(img)
    out_img_y = out.cpu().data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    return out_img
    
if __name__=="__main__":

    model = Net(upscale_factor = Config.upscale_factor)
    model = model.cuda()
    model.load_state_dict(torch.load(Config.model_dir + Config.cnn_model))

    img_name = "noodle.jpg"
    img_path = Config.image_dir + img_name

    out_img = espcn_img(img_path, model)
    out_img.save(Config.result_dir + img_name)
