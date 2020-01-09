'''
This file is to find out the patch image with large bbox, rotate them and then save them. 
The output result can be seen under the folder 'big_rotate_object'.

author: Sifang Fang, Hongfeng Ai
date: 2020-1-7

The following things maybe helpful to understand the code.
1. When sliding on original large bbox, we use 'stride=400'. The large it is, the more patches
   with large bboxes you will get.
2. We define 'x2-x1>900 or y2-y1>900' as the criteria of large bbox based on 1000*1000 image size.
3. Every patch are either clockwise rotated or counter-clockwise rotated with 90 degree.

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
import tqdm

# =========== Setting ===============
train_paths = glob.glob('train/*.json')
save_dir = 'big_rotate_object/'
size = 1000 # image size
stride = 400
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(save_dir+'train/')
    os.makedirs(save_dir+'label/')

# Saving the patch images with large bbox with their labels
for train_path in tqdm.tqdm(train_paths):

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
        
        have_big_object = False
        
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
                        if x2-x1>900 or y2-y1>900:
                            have_big_object = True
                if len(slip_labels)!=0 and have_big_object == True:
                    cv2.imwrite(save_dir+'train/'+Id+'_{}_large.jpg'.format(i),slip_img)
                    with open(save_dir+'label/'+Id+'_{}_large.json'.format(i),'w') as f:
                        json.dump(slip_labels,f)
                    have_big_object = False
                    i += 1


def rotatation(original_x, original_y, original_w, original_h, right_left, patch_size):
    '''
    FUNCTION:
        rotate coordinate label and jpg image
        
    INPUTS:
        1. original_x (int): label - x_min
        2. original_y (int): label - y_min
        3. original_w (int): label - w
        4. original_h (int): label - h
        5. right_left (int): the rotation direction. 1 is the clockwise rotation
        6. patch_size (int): the patch length/height. default 1000
        
    RETURNS:
        new label coordinate
    '''
    label_dicts = []
    if right_left == 1:            
        rotate_x = original_y
        rotate_y = patch_size - original_x - original_w
        rotate_w = original_h
        rotate_h = original_w            
    else:
        rotate_x = patch_size - original_y - original_h
        rotate_y = original_x 
        rotate_w = original_h
        rotate_h = original_w
    
    return rotate_x, rotate_y, rotate_w, rotate_h

# Saving the rotated patch images with large bbox with their labels
input_image_dir = 'big_rotate_object/train/'
input_label_dir = 'big_rotate_object/label/'
save_image_dir = 'big_rotate_object/rotate_image/'
save_label_dir = 'big_rotate_object/rotate_label/'

if not os.path.exists(save_image_dir):
    os.makedirs(save_image_dir)
if not os.path.exists(save_label_dir):
    os.makedirs(save_label_dir)
                
label_file_paths = glob.glob(input_label_dir + '*')
for label_file_path in tqdm.tqdm(label_file_paths):
    Id = label_file_path.split('/')[-1].split('.')[0]
    image_file_path = 'big_rotate_object/train/' + Id + '.jpg'
    
    # read image and label
    img = cv2.imread(image_file_path) 
    with open(label_file_path,'r') as f:
        origin_label_dicts = json.load(f)
    label_dicts = []
    
    # rotation: 1 counter-clockwise rotate；-1 clockwise rotate
    right_left = random.sample((-1,1),1)[0]
    
    # rotate labels
    for origin_label_dict in origin_label_dicts:
        rotate_x, rotate_y, rotate_w, rotate_h = rotatation(origin_label_dict['x'],
                                                            origin_label_dict['y'],
                                                            origin_label_dict['w'],
                                                            origin_label_dict['h'],
                                                            right_left, size)
        label_dicts.append({'x':rotate_x,'y':rotate_y,'w':rotate_w,'h':rotate_h,
                            'class':origin_label_dict['class']})    
    # rotate patch image
    img = np.rot90(img, right_left)         
    
    # save rotated image and label
    cv2.imwrite(save_image_dir+Id+'.jpg', img)
    with open(save_label_dir+Id+'.json', 'w') as outfile:
        json.dump(label_dicts, outfile)


# # ========visualize rotated result==============
# import matplotlib.pyplot as plt

# # random a display image
# img_paths = glob.glob(save_image_dir + '*.jpg')
# img_no = int(random.random()*len(img_paths))
# img_name = img_paths[img_no].split('/')[-1][:-4]

# # read image
# img_arr = cv2.imread(save_image_dir + img_name + '.jpg')

# # read label
# with open(save_label_dir + img_name + '.json') as f:
#     img_labels = json.load(f)

# # add bbox into the image
# for img_label in img_labels:
#     x = img_label['x']
#     y = img_label['y']
#     w = img_label['w']
#     h = img_label['h']
#     cv2.rectangle(img_arr, (x,y),(x+w-1,y+h-1),(0,255,0),5)
#     print(img_label)

# # display image with bbox
# fig = plt.figure(figsize=(14,14))    
# plt.imshow(img_arr)
# plt.title(img_name)
# plt.show()