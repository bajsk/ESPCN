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

    return img
    # return img.unsqueeze(0)

import torchvision.transforms as transforms
RandomColorJitter = transforms.Lambda(

    lambda x: transforms.ColorJitter(brightness = 0.1, contrast = 0.1, hue = 0.01)(x) if random.random() < 0.5 else x)

import random

RandomZoom = transforms.Lambda(

    lambda x: transforms.Resize((224, 224), 2)(transforms.CenterCrop((220, 220))(x)) if random.random() < 0.5 else x)

tmp_preprocess = transforms.Compose([
    RandomZoom,
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees = 10),
    RandomColorJitter,    
])

Normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )  
final_preprocess = transforms.Compose([
    transforms.ToTensor(),
    Normalize
])

if __name__=="__main__":

    cls_model = torchvision.models.resnet50(pretrained = True).cuda().eval()

    espcn_model = ESPCN(upscale_factor = Config.upscale_factor)
    espcn_model = espcn_model.cuda()
    espcn_model.load_state_dict(torch.load(Config.model_dir + Config.cnn_model))
    # espcn_model = None
    
    img_path_list = [Config.image_dir + "/roi_" + str(i) + ".jpg" for i in range(9)]
    img_list = [transform_img(img_path, espcn_model) for img_path in img_path_list]    

    img_list_2 = []
    
    for i in range(10):
        for j, img in enumerate(img_list):
            tmp = tmp_preprocess(img)
            tmp = final_preprocess(tmp)
            img_list_2.append(tmp)
            
    for j, img in enumerate(img_list):
        img_list_2.append(final_preprocess(img))

    img_list = img_list_2
    img_list = [img.unsqueeze(0) for img in img_list]
    
    instance_img_path = Config.image_dir + "/fast_mask_roi_10.jpg"
    # instance_img_path = Config.image_dir + "/noodle.jpg"    
    instance_img_torch = final_preprocess(transform_img(instance_img_path, espcn_model)).unsqueeze(0)
    instance_img_torch = Variable(instance_img_torch).cuda()
    
    # instance_img_torch = Variable(transform_img(instance_img_path, espcn_model)).cuda()
    instance_output = cls_model(instance_img_torch)
    instance_output = F.normalize(instance_output, p = 2, dim = 1)

    h1 = instance_output.cpu().data.numpy()[0].reshape(1, -1)

    feature_list = []
    
    import cv2
    for i, img in enumerate(img_list):
        img_torch = Variable(img).cuda()
        output = cls_model(img_torch)
        output = F.normalize(output, p = 2, dim = 1)
        # euclidean_distance = F.pairwise_distance(output, instance_output)
        # print i, euclidean_distance
        
        feature_list.append(output.cpu().data.numpy().flatten())

    feature_list = np.array(feature_list)
    print feature_list.shape
    classes = [0] * len(img_list)
    classes = np.array(classes, dtype=int)

    import sklearn.svm
    from sklearn.neighbors import NearestNeighbors
    # model = NearestNeighbors(n_neighbors = 5, algorithm = "ball_tree", n_jobs = 4)
    model = NearestNeighbors(n_neighbors = 5, algorithm = "ball_tree", n_jobs = 4, leaf_size = 5)    
    model.fit(feature_list)
    
    # d, i = model.kneighbors(feature_list)
    d2, i2 = model.kneighbors(instance_output.cpu().data.numpy().flatten())
    print d2
    # print model.predict(feature_list)
    # print model.predict(instance_output.cpu().data.numpy().flatten())
