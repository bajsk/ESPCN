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

import rospy
import cv_bridge
from find_similar_objects.srv import *
from find_similar_objects.msg import InhandRoiArray

from sklearn.neighbors import NearestNeighbors
from test_sklearn import load_single_patch, load_augmented_patches, get_single_patch_feature
from fast_mask_segmentation.msg import FastMaskBB2D

def transform_img(img, espcn_model = None):

    if espcn_model is not None:

        img = img.convert('YCbCr')
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

    img = Config.size_preprocess(img)

    return img

def load_single_patch(img, espcn_model = None):

    instance_img_torch = Config.to_tensor_preprocess(transform_img(img, espcn_model)).unsqueeze(0)
    return instance_img_torch

def get_single_patch_feature(img, espcn_model = None):

    instance_img_torch = load_single_patch(img, espcn_model)
    instance_img_torch = Variable(instance_img_torch).cuda()
    instance_output = cls_model(instance_img_torch)
    instance_output = F.normalize(instance_output, p = 2, dim = 1)

    return instance_output

class FindSimilarObjectsServer():

    def __init__(self, gpu_id = None):

        self.gpu_id = gpu_id
        self.cls_model = None
        self.espcn_model = None
        self.br = cv_bridge.CvBridge()
        self.model = NearestNeighbors(n_neighbors = 5, algorithm = "ball_tree", metric = "minkowski", n_jobs = 4, leaf_size = 5, p = 2)
        s = rospy.Service("find_similar_objects_server", FindSimilarRois, self.handle_similar_objects)
        rospy.spin()

    def handle_similar_objects(self, req):
        
        if self.cls_model == None and self.espcl_model == None:
            try:
                self.cls_model = torchvision.models.resnet50(pretrained = True).cuda().eval()
                self.espcn_model = ESPCN(upscale_factor = Config.upscale_factor)
                self.espcn_model = self.espcn_model.cuda()
                self.espcn_model.load_state_dict(torch.load(Config.model_dir + Config.cnn_model))
            except:
                rospy.logerr("Error, cannot load cls and espcn net to the GPU")
                self.cls_model = None
                self.espcl_model = None
                self.service_queue = -1
                return FindSimilarRoisResponse()

        try:

            img_list = []
            for i, img_br in enumerate(req.inhand_obj_rois.inhand_roi_arr):
                cv_img = self.br.imgmsg_to_cv2(img_br, desired_encoding = "bgr8")
                pil_img = Image.fromarray(cv_img)
                img_list.append(transform_img(pil_img))

            img_list = load_augmented_patches(img_list)

            feature_list = []

            for i, img in enumerate(img_list):

                img_torch = Variable(img).cuda()
                output = self.cls_model(img_torch)
                output = F.normalize(output, p = 2, dim = 1)
                feature_list.append(output.cpu().data.numpy().flatten())

            feature_list = np.array(feature_list)
            self.model.fit(feature_list)

            inshelf_img = self.br.imgmsg_to_cv2(req.inshelf_img, desired_encoding = "bgr8")
            pil_inshelf_img = Image.fromarray(inshelf_img)

            matching_distance = 9999
            matching_x = None
            matching_y = None
            matching_w = None
            matching_h = None
            
            for i, _bb in enumerate(inshelf_obj_rois.fm_bbox_arr):
                        
                x = int(_bb.bbox.x)
                y = int(_bb.bbox.y)
                w = int(_bb.bbox.w)
                h = int(_bb.bbox.h)   

                pil_inshelf_roi_img = pil_inshelf_img.crop((x, y, width, height))
                instance_output = get_single_patch_feature(instance_img_path, self.espcn_model)
                d, _i = model.kneighbors(instance_output.cpu().data.numpy().reshape(1, -1))
                distance_mean = d.mean()

                if distance_mean < matching_distance:
                    matching_distance = distance_mean
                    matching_x = x
                    matching_y = y
                    matching_w = w
                    matching_h = h

            matching_box = FastMaskBB2D()
            matching_box.bbox.x = matching_x
            matching_box.bbox.y = matching_y
            matching_box.bbox.w = matching_w
            matching_box.bbox.h = matching_h
            
            return FindSimilarRoisResponse(matching_roi = matching_box)

        except cv_bridge.CvBridgeError as e:
            rospy.logerr("CvBridge exception %s", e)
            return FindSimilarRoisResponse()

if __name__ == "__main__":

    rospy.init_node("find_similar_objects_server")
    FindSimilarObjectsServer(gpu_id = 0)
