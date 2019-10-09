#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:34:52 2019

@author: paddle
"""

import os
import cv2

ls = os.listdir('./infer_result')

for i in ls:
    temp = cv2.imread('./infer_result/'+i)
    temp = cv2.resize(temp, (480,360))
    cv2.imwrite('./infer_result/'+i, temp)









