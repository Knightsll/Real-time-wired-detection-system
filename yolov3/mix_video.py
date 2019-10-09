#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:56:42 2019

@author: paddle
"""

import numpy as np
import cv2
from skvideo.io import FFmpegWriter
from PIL import Image

fps = 20
video_dir = './output_frame/detected.mp4'
num = 1020

 

video_writer = FFmpegWriter(video_dir)

for i in range(num):
    if i%100==0:
        print('{}%         '.format(i/num*100)+'%d'%i)
    frame = cv2.imread('/home/paddle/Desktop/Baidu_AI/yolov3/output_frame/{0}.png'.format(i))
    frame = frame[:,:,::-1]
    frame = cv2.resize(frame, (492,369))
    
    video_writer.writeFrame(frame)
    
video_writer.close()
print('finish')

