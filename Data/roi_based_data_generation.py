'''
This file is to generate data.
The data generation uses roi-based sliding method.

author: Sifang Fang
date: 2019-12-17

cd Data
'''

import sys
sys.path.append('/home/admin/jupyter/kfbreader-linux')
import kfbReader
import json
import os
import cv2
import numpy as np
import random
import math
import glob


train_paths = glob.glob('train/*.json')
save_dir = 'Train/'
size = 1000
stride = 500
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(save_dir+'train/')
    os.makedirs(save_dir+'label/')
for train_path in train_paths:

    with open(train_path,'r') as f:
        train_dicts = json.load(f)
    Id = train_path.split('/')[-1].split('.')[0]
    kfb_path = os.path.join('train',Id+'.kfb')
    read = kfbReader.reader()
    read.ReadInfo(kfb_path, 20, True)
    
    rois = []
    for train_dict in train_dicts:
        if train_dict['class']=='roi':
            rois.append(train_dict)
    i = 0
    for roi in rois:

        X,Y,W,H = roi['x'],roi['y'],roi['w'],roi['h']
        W = int(math.ceil(W/stride)*stride)
        H = int(math.ceil(H/stride)*stride)
        img = read.ReadRoi(X, Y, W, H, 20).copy()
        label_dicts = []
        for train_dict in train_dicts:
            if train_dict['class'] is not 'roi':
                x,y,w,h = train_dict['x'],train_dict['y'],train_dict['w'],train_dict['h']
                if X<x<X+W and Y<y<Y+H:
                    label_dicts.append({'x':x,'y':y,'w':w,'h':h,'class':train_dict['class']})
        stride_x = W//stride - 1
        stride_y = H//stride - 1
        nums = 0
        for sx in range(stride_x):
            for sy in range(stride_y):
                slip_img = img[sy*stride:(sy*stride+size),sx*stride:(sx*stride+size)].copy()
                
                img_w,img_h = slip_img.shape[0],slip_img.shape[1]
                slip_labels = []
                for label_dict in label_dicts:
                    nx = label_dict['x'] - X - sx*stride
                    ny = label_dict['y'] - Y - sy*stride
                    nw = label_dict['w']
                    nh = label_dict['h']
                    x1 = max(nx, 0)
                    y1 = max(ny, 0)
                    x2 = min(img_w,nx+nw)
                    y2 = min(img_h,ny+nh)
                    intersect_area = (x2-x1)*(y2-y1)
                    area = nw*nh
                    if x1 >img_w or y1>img_h or x2<0 or y2<0:
                        continue
                    elif intersect_area>500000 or (intersect_area/area)>0.5:
                        assert x2>x1
                        assert y2>y1
                        slip_labels.append({'x':x1,'y':y1,'w':x2-x1,'h':y2-y1,'class':label_dict['class']})
                if len(slip_labels)==0 and nums<=0:
                    continue
                elif len(slip_labels)==0 and nums>0:
                    nums -= 1
                elif len(slip_labels)>0:
                    nums += 1
                cv2.imwrite(save_dir+'train/'+Id+'_{}.jpg'.format(i),slip_img)
                with open(save_dir+'label/'+Id+'_{}.json'.format(i),'w') as f:
                    json.dump(slip_labels,f)
                i += 1 

# # VISUALIZATION
# if not os.path.exists('visual/'):
#     os.makedirs('visual/')
# for label_path in glob.glob('Train1/label/*.json'):
#     with open(label_path,'r') as f:
#         labels = json.load(f)
#     if len(labels)==0:
#         continue
#     Id = label_path.split('/')[-1].split('.')[0]
#     img_path = os.path.join('Train1','train',Id+'.jpg')
#     img = cv2.imread(img_path)
#     for label in labels:
#         x,y,w,h = label['x'],label['y'],label['w'],label['h']
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
#     cv2.imwrite('visual/'+Id+'_w{}_h{}'.format(w,h)+label['class']+'.jpg',img)

# if not os.path.exists('true_vision/'):
#     os.makedirs('true_vision/')
# with open('train/101.json','r') as f:
#     labels = json.load(f)
# read = kfbReader.reader()
# read.ReadInfo('train/101.kfb', 20, True)
# i=0
# for label in labels:
#     if label['class']=='roi':
#         continue
#     x,y,w,h = label['x'],label['y'],label['w'],label['h']
#     img = read.ReadRoi(x, y, w, h, 20).copy()
#     cv2.imwrite('true_vision/w{}_h{}_'.format(w,h)+label['class']+str(i)+'.jpg',img)
#     i+=1