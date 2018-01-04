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
import cv2

def transform_img(img_path, espcn_model = None):

    if espcn_model is not None:
        img = espcn_img(img_path, espcn_model)
    else:
        img = Image.open(img_path)
        
    img = Config.size_preprocess(img)

    return img

def load_single_patch(img_path, espcn_model = None):

    instance_img_torch = Config.to_tensor_preprocess(transform_img(img_path, espcn_model)).unsqueeze(0)
    return instance_img_torch

def load_augmented_patches(img_list, num_factor = 10):

    aug_img_list = []
    
    for i in range(num_factor):
        for _, img in enumerate(img_list):
            augmented_img = Config.random_data_aug_preprocess(img)
            augmented_img = Config.to_tensor_preprocess(augmented_img)
            aug_img_list.append(augmented_img)

    for _, img in enumerate(img_list):
        aug_img_list.append(Config.to_tensor_preprocess(img))
    
    return [img.unsqueeze(0) for img in aug_img_list]

if __name__=="__main__":

    cls_model = torchvision.models.resnet50(pretrained = True).cuda().eval()

    espcn_model = ESPCN(upscale_factor = Config.upscale_factor)
    espcn_model = espcn_model.cuda()
    espcn_model.load_state_dict(torch.load(Config.model_dir + Config.cnn_model))
    # espcn_model = None
    
    img_path_list = [Config.image_dir + "/roi_" + str(i) + ".jpg" for i in range(9)]
    img_list = [transform_img(img_path, espcn_model) for img_path in img_path_list]    
    img_list = load_augmented_patches(img_list)
    
    instance_img_path = Config.image_dir + "/fast_mask_roi_10.jpg"
    instance_img_path = Config.image_dir + "/fast_mask_roi_11.jpg"    
    instance_img_torch = load_single_patch(instance_img_path, espcn_model)
    instance_img_torch = Variable(instance_img_torch).cuda()
    instance_output = cls_model(instance_img_torch)
    instance_output = F.normalize(instance_output, p = 2, dim = 1)

    feature_list = []
    for i, img in enumerate(img_list):
        img_torch = Variable(img).cuda()
        output = cls_model(img_torch)
        output = F.normalize(output, p = 2, dim = 1)
        feature_list.append(output.cpu().data.numpy().flatten())

    feature_list = np.array(feature_list)

    from sklearn.neighbors import NearestNeighbors
    # model = NearestNeighbors(n_neighbors = 5, algorithm = "ball_tree", n_jobs = 4)
    model = NearestNeighbors(n_neighbors = 5, algorithm = "ball_tree", n_jobs = 4, leaf_size = 5)    
    model.fit(feature_list)
    
    # d, i = model.kneighbors(feature_list)
    d2, i2 = model.kneighbors(instance_output.cpu().data.numpy().reshape(1, -1))
    print d2
