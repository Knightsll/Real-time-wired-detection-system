#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:57:25 2019

@author: paddle
"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure()
_, ax = plt.subplots(1)

rect = plt.Rectangle(
            (0, 0),
            0.5,
            0.5,
            fill=False,
            linewidth=2.0,
            edgecolor='red')
ax.add_patch(rect)
