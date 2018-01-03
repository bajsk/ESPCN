#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import torchvision
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from config import Config

from espcn_model import Net as ESPCN
from espcn_img import espcn_img

from PIL import Image

def transform_img(img_path, espcn_model = None):

    if espcn_model is not None:
        img = espcn_img(img_path, espcn_model)
    else:
        img = Image.open(img_path)
        
    img = Config.preprocess(img)

    return img.unsqueeze(0)

if __name__=="__main__":

    cls_model = torchvision.models.resnet50(pretrained = True).cuda().eval()

    espcn_model = ESPCN(upscale_factor = Config.upscale_factor)
    espcn_model = espcn_model.cuda()
    espcn_model.load_state_dict(torch.load(Config.model_dir + Config.cnn_model))
    
    img_path_list = [Config.image_dir + "/roi_" + str(i) + ".jpg" for i in range(9)]
    img_list = [transform_img(img_path, espcn_model) for img_path in img_path_list]    

    instance_img_path = Config.image_dir + "/fast_mask_roi_10.jpg"
    instance_img_torch = Variable(transform_img(instance_img_path, espcn_model)).cuda()
    instance_output = cls_model(instance_img_torch)
    instance_output = F.normalize(instance_output, p = 2, dim = 1)

    for i, img in enumerate(img_list):
        img_torch = Variable(img).cuda()
        output = cls_model(img_torch)
        output = F.normalize(output, p = 2, dim = 1)
        euclidean_distance = F.pairwise_distance(output, instance_output)
        print euclidean_distance
