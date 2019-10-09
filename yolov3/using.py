#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 22:03:58 2019

@author: paddle
"""

# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import numpy as np
import paddle
import paddle.fluid as fluid
import box_utils
import reader
from utility import print_arguments, parse_args
from models.yolov3 import YOLOv3
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg
import cv2
from skvideo.io import FFmpegWriter
import sys
sys.path.append(r'/home/paddle/Desktop/Baidu_AI/yolov3/')
def infer():

    if not os.path.exists('output_frame'):
        os.mkdir('output_frame')

    model = YOLOv3(is_train=False)
    model.build_model()
    outputs = model.get_pred()
    input_size = cfg.input_size
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if cfg.weights:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.weights, var.name))
        fluid.io.load_vars(exe, cfg.weights, predicate=if_exist)
    # yapf: enable

    # you can save inference model by following code
    # fluid.io.save_inference_model("./output/yolov3", 
    #                               feeded_var_names=['image', 'im_shape'],
    #                               target_vars=outputs,
    #                               executor=exe)

    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())
    fetch_list = [outputs]
    
    
    video_path = "" #source
    video_dir = ""  #output path
    
    
    video_writer = FFmpegWriter(video_dir)
    cap = cv2.VideoCapture(video_path)
    i=0
    while cap.isOpened():
        if i%100==0:
            print('{}%         '.format(i/1020*100)+'%d'%i)
        label_names, _ = reader.get_label_infos()
        ok, image = cap.read()
        if ok:
            infer_reader = reader.infer_m(input_size, image)
            data = next(infer_reader())
            im_shape = data[0][2]
            outputs = exe.run(fetch_list=[v.name for v in fetch_list],
                              feed=feeder.feed(data),
                              return_numpy=False)
            bboxes = np.array(outputs[0])
            if bboxes.shape[1] != 6:
                continue
            labels = bboxes[:, 0].astype('int32')
            scores = bboxes[:, 1].astype('float32')
            boxes = bboxes[:, 2:].astype('float32')
    
            frame =  box_utils.save_as_video(image, boxes, scores, labels, label_names,
                                          cfg.draw_thresh)
            frame = frame[:,:,::-1]
            frame = cv2.resize(frame, (492,369))
            
            video_writer.writeFrame(frame)
        else:
            print("over")
            break
        i+=1
    video_writer.close()
        

def call_infer():

    if not os.path.exists('output_frame'):
        os.mkdir('output_frame')

    model = YOLOv3(is_train=False)
    model.build_model()
    outputs = model.get_pred()
    input_size = cfg.input_size
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if cfg.weights:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.weights, var.name))
        fluid.io.load_vars(exe, cfg.weights, predicate=if_exist)
    # yapf: enable

    # you can save inference model by following code
    # fluid.io.save_inference_model("./output/yolov3", 
    #                               feeded_var_names=['image', 'im_shape'],
    #                               target_vars=outputs,
    #                               executor=exe)

    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())
    fetch_list = [outputs]
    
    
    video_path = cfg.video_path
    video_dir = cfg.video_dir
    
    
    video_writer = FFmpegWriter(video_dir)
    cap = cv2.VideoCapture(video_path)
    i=0
    while cap.isOpened():
        if i%100==0:
            print('{}s video done       '.format(i/30))
        label_names, _ = reader.get_label_infos()
        ok, image = cap.read()
        if ok:
            infer_reader = reader.infer_m(input_size, image)
            data = next(infer_reader())
            im_shape = data[0][2]
            outputs = exe.run(fetch_list=[v.name for v in fetch_list],
                              feed=feeder.feed(data),
                              return_numpy=False)
            bboxes = np.array(outputs[0])
            if bboxes.shape[1] != 6:
                continue
            labels = bboxes[:, 0].astype('int32')
            scores = bboxes[:, 1].astype('float32')
            boxes = bboxes[:, 2:].astype('float32')
            
            frame =  box_utils.save_as_video(image, boxes, scores, labels, label_names,
                                          cfg.draw_thresh)
            """
            frame = frame[:,:,::-1]
            frame = cv2.resize(frame, (int(500*1.2),int(350*1.2)))
            video_writer.writeFrame(frame)
            """
            cv2.imshow("Video Detector", frame)
            cv2.waitKey(1)
        else:
            print("over")
            break
        i+=1
    video_writer.close()
        
if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    call_infer()
