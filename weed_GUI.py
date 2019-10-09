#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:50:55 2019

@author: paddle
"""

import cv2
import tkinter as tk
import os
from tkinter import filedialog#文件控件
from PIL import Image, ImageTk#图像控件
import threading#多线程
import time
import subprocess as sub
import cv2
import sys
sys.path.append('/home/paddle/Desktop/Baidu_AI/yolov3/')


#---------------创建窗口
window = tk.Tk()
window.title('Weed Detector')
sw = window.winfo_screenwidth()#获取屏幕宽
sh = window.winfo_screenheight()#获取屏幕高
wx = 1000
wh = 800

window.geometry("%dx%d+%d+%d" %(wx,wh,(sw-wx)/2,(sh-wh)/2-100))#窗口至指定位置
canvas = tk.Canvas(window,bg='#c4c2c2',height=wh,width=wx)#绘制画布


#---------------打开摄像头获取图片
def video_demo():
    def cc():
        capture = cv2.VideoCapture('/home/paddle/Desktop/Baidu_AI/yolov3/frame/full.mp4')
        while True:
            ret, frame = capture.read()#从摄像头读取照片
            #frame = cv2.flip(frame, 1)#翻转 0:上下颠倒 大于0水平颠倒
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            image_file=ImageTk.PhotoImage(img)
            canvas.create_image(0,0,anchor='nw',image=image_file)
            time.sleep(1/35)
    t=threading.Thread(target=cc)
    t.start()

def video_test():
    def cc():
        capture = cv2.VideoCapture('/home/paddle/Desktop/Baidu_AI/yolov3/frame/full.mp4')
        while True:
            ret, frame = capture.read()#从摄像头读取照片
            #frame = cv2.flip(frame, 1)#翻转 0:上下颠倒 大于0水平颠倒
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            image_file=ImageTk.PhotoImage(img)
            canvas.create_image(0,0,anchor='nw',image=image_file)
            time.sleep(1/35)
    t=threading.Thread(target=cc)
    t.start()

def video_choose():
    a=tk.filedialog.askopenfilename()
    print(a)
    def cc(path):
        capture = cv2.VideoCapture(path)
        while True:
            ret, frame = capture.read()#从摄像头读取照片
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            image_file=ImageTk.PhotoImage(img)
            canvas.create_image(0,0,anchor='nw',image=image_file)
            time.sleep(1/35)
    t=threading.Thread(target=lambda:cc(a))
    t.start()


def train_operator():
    
    def sure(batch_size,data_path,pretrain,output,learning_rate):
        os.system('cd yolov3 && python train.py --model_save_dir {0} --pretrain {1} '.format(output,pretrain)+
                  '--use_multiprocess False --data_dir {} --class_num 4 '.format(data_path)+ 
                  '--batch_size {0} --learning_rate {1}'.format(batch_size,learning_rate))
    #Training handle
    window_son = tk.Toplevel()
    window_son.geometry('600x400')
    window_son.title('Training Operation Parameter setting')
    
    tk.Label(window_son, text = "Current folder path : Baidu_AI/ylov3",font=("Time New Roman",20)).place(x=10, y=5)
    
    #Batch size setting
    batch_size = tk.IntVar()
    batch_size.set(8)
    tk.Label(window_son, text = "Batch size: ",font=("Time New Roman",18)).place(x=10, y=60)
    entry_new_name = tk.Entry(window_son, textvariable=batch_size,font=("Time New Roman",18))
    entry_new_name.place(x=260, y=60)
    
    #data path setting
    data_path = tk.StringVar()
    data_path.set('./dataset/weed/')
    tk.Label(window_son, text = "Data path: ",font=("Time New Roman",18)).place(x=10, y=110)
    entry_data_path = tk.Entry(window_son, textvariable=data_path,font=("Time New Roman",18))  
    entry_data_path.place(x=260, y=110)  

    #pretrained weights setting
    pre_path = tk.StringVar()
    pre_path.set('./weights/yolov3')
    tk.Label(window_son, text = "Pretrain: ",font=("Time New Roman",18)).place(x=10, y=160)
    entry_pre_path = tk.Entry(window_son, textvariable=pre_path,font=("Time New Roman",18))  
    entry_pre_path.place(x=260, y=160)  
    
    #output model path setting
    output_path = tk.StringVar()
    output_path.set('./weed_output')
    tk.Label(window_son, text = "Output Path: ",font=("Time New Roman",18)).place(x=10, y=210)
    entry_output_path = tk.Entry(window_son, textvariable=output_path,font=("Time New Roman",18))  
    entry_output_path.place(x=260, y=210)  
    
    #Learning rate setting
    learning_rate = tk.DoubleVar()
    learning_rate.set(0.001)
    tk.Label(window_son, text = "Learning rate: ",font=("Time New Roman",18)).place(x=10, y=260)
    entry_learning_rate = tk.Entry(window_son, textvariable=learning_rate,font=("Time New Roman",18))  
    entry_learning_rate.place(x=260, y=260)  
    
    su = tk.Button(window_son, text = 'Sure',font=("Time New Roman",15), height = 3, width=15, command = lambda: sure(batch_size.get(),data_path.get(),pre_path.get(),output_path.get(),learning_rate.get()))
    su.place(x = 10, y = 310)
    
    qu = tk.Button(window_son, text = 'Quit',font=("Time New Roman",15), height = 3, width=15, command = window_son.destroy)
    qu.place(x = 300, y = 310)


def transform():
    #Training handle
    window_son = tk.Toplevel()
    window_son.geometry('600x400')
    window_son.title('Video detector setting')
    
    tk.Label(window_son, text = "Current folder path : Baidu_AI/ylov3",font=("Time New Roman",20)).place(x=10, y=5)
    def sure(weight,video, output_path):
        print("cd yolov3/ && python using.py  --video_path {0}  --video_dir {1} --weights {2} --data_dir dataset/weed --class_num 4".format(video,output_path,weight))
        os.system("cd yolov3/ && python using.py  --video_path {0}  --video_dir {1} --weights {2} --data_dir dataset/weed --class_num 4".format(video,output_path,weight))
        os.system("cd ..")
    
    def play():
        a=tk.filedialog.askopenfilename()
        print(a)
        def cc(path):
            capture = cv2.VideoCapture(path)
            while True:
                ret, frame = capture.read()
                frame = cv2.resize(frame, (int(500*1.2),int(350*1.2)))
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                image_file=ImageTk.PhotoImage(img)
                canvas.create_image(200,100,anchor='nw',image=image_file)
                time.sleep(1/35)
        t=threading.Thread(target=lambda:cc(a))
        t.start()
        

    
    #Batch size setting
    weight = tk.StringVar()
    weight.set('weed_output/model_iter49999')
    tk.Label(window_son, text = "Weight: ",font=("Time New Roman",18)).place(x=10, y=70)
    entry_weight = tk.Entry(window_son, textvariable=weight,font=("Time New Roman",18))  
    entry_weight.place(x=260, y=70)  
    
    #data path setting
    output_path = tk.StringVar()
    output_path.set('../Video/1_detected.mp4')
    tk.Label(window_son, text = "Video output path: ",font=("Time New Roman",18)).place(x=10, y=140)
    entry_output_path = tk.Entry(window_son, textvariable=output_path,font=("Time New Roman",18))  
    entry_output_path.place(x=260, y=140)  
    
    video_path = tk.StringVar()
    video_path.set('../Video/1.mp4')
    tk.Label(window_son, text = "Source path: ",font=("Time New Roman",18)).place(x=10, y=210)
    entry_video_path = tk.Entry(window_son, textvariable=video_path,font=("Time New Roman",18))  
    entry_video_path.place(x=260, y=210)  
    

    su = tk.Button(window_son, text = 'Sure',font=("Time New Roman",15), height = 3, width=16, command = lambda:sure(weight.get(),video_path.get(), output_path.get()))
    su.place(x = 10, y = 280)
    
    py = tk.Button(window_son, text = 'Play',font=("Time New Roman",15), height = 3, width=16, command =play)
    py.place(x = 220, y = 280)
    
    qu = tk.Button(window_son, text = 'Quit',font=("Time New Roman",15), height = 3, width=16, command =window_son.destroy)
    qu.place(x = 430, y = 280)
    
def Data_Enhanced():
    #Training handle
    window_son = tk.Toplevel()
    window_son.geometry('600x400')
    window_son.title('Data Enhanced Setting')
    
    tk.Label(window_son, text = "Current folder path : Baidu_AI/PaddleGAN",font=("Time New Roman",20)).place(x=10, y=5)
    def sure(batch_size,data_dir, output_path):
        print("cd PaddleGAN/ && python train.py  --dataset {1} --model_net CycleGAN --net_G resnet_9block --batch_size {0} --net_D basic --norm_type batch_norm --epoch 100 --output {2} --crop_type Random --load_size 286 --crop_size 256".format(batch_size,data_dir,output_path))
        os.system("cd PaddleGAN/ && python train.py  --dataset {1} --model_net CycleGAN --net_G resnet_9block --batch_size {0} --net_D basic --norm_type batch_norm --epoch 100 --output {2} --crop_type Random --load_size 286 --crop_size 256".format(batch_size,data_dir,output_path))
        os.system("cd ..")
    
    def change(model_choose, enhance_path):
        print("cd PaddleGAN/ && python infer.py --init_model {} --dataset_dir {} --image_size 256 --input_style A --model_net CycleGAN --net_G resnet_9block  --g_base_dims 32".format(model_choose,enhance_path))
        os.system("cd PaddleGAN/ && python infer.py --init_model {} --dataset_dir '{}' --image_size 256 --input_style A --model_net CycleGAN --net_G resnet_9block  --g_base_dims 32".format(model_choose, enhance_path))
        os.system("cd ..")
        path = '/home/paddle/Desktop/Baidu_AI/PaddleGAN/infer_result/'
        ls = os.listdir(path)
        for i in ls:
            temp = cv2.imread(path+i)
            temp = cv2.resize(temp, (480,360))
            cv2.imwrite(path+i, temp)
        
    def play():
        a=tk.filedialog.askopenfilename()
        print(a)
        def cc(path):
            capture = cv2.VideoCapture(path)
            while True:
                ret, frame = capture.read()
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                image_file=ImageTk.PhotoImage(img)
                canvas.create_image(200,100,anchor='nw',image=image_file)
                time.sleep(1/35)
        t=threading.Thread(target=lambda:cc(a))
        t.start()
        

    
    #Batch size setting
    batch_size = tk.IntVar()
    batch_size.set(8)
    tk.Label(window_son, text = "Batch size: ",font=("Time New Roman",18)).place(x=10, y=60)
    entry_new_name = tk.Entry(window_son, textvariable=batch_size,font=("Time New Roman",18))  
    entry_new_name.place(x=280, y=60)  
    
    #data path setting
    data_dir = tk.StringVar()
    data_dir.set('./data/light')
    tk.Label(window_son, text = "Data path: ",font=("Time New Roman",18)).place(x=10, y=110)
    entry_data_path = tk.Entry(window_son, textvariable=data_dir,font=("Time New Roman",18))  
    entry_data_path.place(x=280, y=110)  

    #pretrained weights setting
    output_path = tk.StringVar()
    output_path.set('./model')
    tk.Label(window_son, text = "Model Output: ",font=("Time New Roman",18)).place(x=10, y=160)
    entry_pre_path = tk.Entry(window_son, textvariable=output_path,font=("Time New Roman",18))  
    entry_pre_path.place(x=280, y=160)  
    
    #output model path setting
    enhance_path = tk.StringVar()
    enhance_path.set('./data/change_light/testA/*')
    tk.Label(window_son, text = "Required data Path: ",font=("Time New Roman",18)).place(x=10, y=210)
    entry_output_path = tk.Entry(window_son, textvariable=enhance_path,font=("Time New Roman",18))  
    entry_output_path.place(x=280, y=210)  
    
    model_choose = tk.StringVar()
    model_choose.set('./light_output_sun/checkpoints/99/')
    tk.Label(window_son, text = "Enhanced model choose: ",font=("Time New Roman",18)).place(x=10, y=260)
    entry_output_path = tk.Entry(window_son, textvariable=model_choose,font=("Time New Roman",18))  
    entry_output_path.place(x=280, y=260) 
    

    su = tk.Button(window_son, text = 'Train',font=("Time New Roman",15), height = 3, width=16, command = lambda:sure(batch_size.get(),data_dir.get(), output_path.get()))
    su.place(x = 10, y = 310)
    
    py = tk.Button(window_son, text = 'Enhance',font=("Time New Roman",15), height = 3, width=16, command =lambda: change(model_choose.get(),enhance_path.get()))
    py.place(x = 220, y = 310)
    
    qu = tk.Button(window_son, text = 'Quit',font=("Time New Roman",15), height = 3, width=16, command =window_son.destroy)
    qu.place(x = 430, y = 310)


def back_select():

    path = tk.filedialog.askopenfilename()
    load = Image.open(path)
    render = ImageTk.PhotoImage(load)
 
    img = tk.Label(image=render)
    img.image = render
    img.place(x=0, y=0)

imgpath = '/home/paddle/Desktop/Baidu_AI/resource/fire.gif'
img = Image.open(imgpath)
photo = ImageTk.PhotoImage(img)
canvas.create_image(0, 0, image=photo,anchor = 'nw')
canvas.pack()

bt_start = tk.Button(window,text='Training Operation',font=("Time New Roman",10),height=2,width=15,command=train_operator)
bt_start.place(x=120,y=710)

rt_start = tk.Button(window,text='Data Enhance',font=("Time New Roman",10),height=2,width=15,command=Data_Enhanced)
rt_start.place(x=330,y=710)

gt_start = tk.Button(window,text='Video Detector',font=("Time New Roman",10),height=2,width=15,command=transform)
gt_start.place(x=540,y=710)

qt = tk.Button(window, text = 'Quit',font=("Time New Roman",10), height = 2, width=15, command = window.quit)
qt.place(x=750,y = 710)

window.mainloop()



temp = 'python infer.py --init_model light_output_sun/checkpoints/99/ --dataset_dir "data/change_light/testA/*" --image_size 256 --input_style A --model_net CycleGAN --net_G resnet_9block  --g_base_dims 32'


