#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 22:35:33 2019

@author: paddle
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


file_path = '/home/paddle/Desktop/Baidu_AI/yolov3/dataset/coco/test.mp4'
catelogy = '/home/paddle/Desktop/Baidu_AI/yolov3/frame/'
cap = cv2.VideoCapture(file_path)
i=0
while True:
    ok, img = cap.read()
    print(np.shape(img))
    cv2.imwrite(catelogy+'{0}.jpg'.format(i), img)
    i+=1


