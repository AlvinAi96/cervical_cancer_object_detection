'''
This file is to check data distribution and generate data.
The data generation uses pos-center-based sliding method.

author: Hongfeng Ai
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
import matplotlib.pyplot as plt
import copy

def check_data_distribution(json_paths):
    '''
    FUNCTION:  check the data distribution among different classes.
                - small bbox: area < 32*32
                - medium bbox: 32*32 <= area < 96*96
                - large bbox: area > 96*96
    
    INPUTS:
        json_paths (list): the paths to all json label files.
    
    RETURNS:
        print out the data distribution result.
    '''
    # record the number of different kinds of poses (i.e. bboxes)
    pos_num_dict = {}

    for c in CLASS_NAMES:
        small_pos_num = 0
        medium_pos_num = 0
        large_pos_num = 0

        for p in json_paths:
            
            # load train json file to gain labels
            with open(p, 'r') as f:
                train_dicts = json.load(f)
            
            for train_dict in train_dicts:
                if train_dict['class'] == c:
                    w = train_dict['w']
                    h = train_dict['h']
                    area = w * h

                    # accumulate the number of different sizes of poses
                    if area < 32**2:
                        small_pos_num += 1
                    elif area > 96**2:
                        large_pos_num += 1
                    else:
                        medium_pos_num += 1

        pos_num_dict[c] = [small_pos_num, medium_pos_num, large_pos_num]
        print('Class-%s: small: %d, medium: %d, large: %d' % (c, small_pos_num, medium_pos_num, large_pos_num))


def iou_filter(new_roi_label, all_bbox_labels, iou_thred=0.5):
    '''
    FUNCTION: Based on the IOU beween the roi assigned randomly (i.e., patch) and other bboxes,
              rewrite new bbox labels for the patch.
              result format: [{'x', 'y', 'w', 'h'}, {}, ...]

    INPUTS:
        1. new_roi_label (dict): a dict contains a patch coordinate information. e.g.,{xywh}
        2. all_bbox_labels (list): a list contains all bboxes coordinate information of a kfb image. e.g.,[{xywh},{xywh},...]
        3. iou_thred (float): only count the IOU area as new bbox when IOU >= iou_thred
    
    RETURNS:
        new_bbox_labels (list): a list of new bbox coordinate information for the assigned patch. 
    '''
    # new patch coordinate information
    roi_xmin = new_roi_label['x']
    roi_ymin = new_roi_label['y']
    roi_w = new_roi_label['w']
    roi_h = new_roi_label['h']

    new_bbox_labels = []

    for bbox_label in all_bbox_labels:
        if bbox_label['class'] != 'roi':
            # bbox coordinate information
            bbox_xmin = bbox_label['x']
            bbox_ymin = bbox_label['y']
            bbox_w = bbox_label['w']
            bbox_h = bbox_label['h']

            # get intersection:
            if bbox_w * bbox_h < IMG_SIZE**2 and bbox_w < IMG_SIZE and bbox_h < IMG_SIZE:
                x1 = max(bbox_xmin, roi_xmin)
                y1 = max(bbox_ymin, roi_ymin)
                x2 = min(bbox_xmin + bbox_w - 1, roi_xmin + roi_w - 1)
                y2 = min(bbox_ymin + bbox_h -1, roi_ymin + roi_h - 1)
            
                w = np.maximum(0, x2 - x1 + 1)    # the width of overlap
                h = np.maximum(0, y2 - y1 + 1)    # the height of overlap 
                
                iou_area = w * h
                bbox_area = bbox_w * bbox_h

                # iou_area > 0
                if (iou_area / bbox_area) > iou_thred:
                    new_bbox_labels.append({'x':x1-roi_xmin, 'y':y1-roi_ymin, 'w':x2-x1+1, 'h':y2-y1+1, 'class':bbox_label['class']})
            else:
                pass
    
    return new_bbox_labels


def produce_patch_for_large_bbox(large_bbox_labels, img_name, stride_proportion, img_size, read_tool, save_dir):
    '''
    FUNCTION:
        produce patch images and their coordinate json file for large bbox (i.e., > IMG_SIZE**2)
    
    INPUTS:
        1. large_bbox_labels (list): a list of large bbox labels. e.g., [{xywh},{}].
        2. img_name (str): the image name.
        3. stride (float): the sliding patch stride.
        4. img_size (int): image height = length
        5. read_tool (func): kfbreader.
        6. save_dir (str): the saving path.

    RETURNS:
        save patch jpg and json files, and return the number of new patches for large bboxes.
    '''
    
    stride = img_size * stride_proportion

    fully_positive_patch_nums = 0    
    for large_bbox_label in large_bbox_labels:
        # get the horizontal/vertical moving step (i.e., sliding step) 
        hor_move_step = int((large_bbox_label['w'] - img_size)/stride + 1)
        ver_move_step = int((large_bbox_label['h'] - img_size)/stride + 1)
        
        if hor_move_step < 1:
            hor_move_step = 1
        if ver_move_step < 1:
            ver_move_step = 1

        # get the start sliding coordinate information
        x_start = int(large_bbox_label['x'])
        y_start = int(large_bbox_label['y'])
        width = int(copy.deepcopy(img_size))
        height = int(copy.deepcopy(img_size))

        # two forloops for horizontal and vertical moving the windows
        for ver in range(ver_move_step):
            for hor in range(hor_move_step):
                x_new = int(x_start + ver * stride)
                y_new = int(y_start + hor * stride)

                # get sliding patch array (1000*1000)
                img_arr = read_tool.ReadRoi(x_new, y_new, width, height, 20)  
                
                # different large bbox situations have to be used different strategy to 
                # correct the coordinate information on new patch image
                if large_bbox_label['w'] > width and large_bbox_label['h'] < height:

                    w_correct = width
                    h_correct = large_bbox_label['h']
                elif large_bbox_label['w'] < width and large_bbox_label['h'] > height:
                    w_correct = large_bbox_label['w']
                    h_correct = height
                else:
                    w_correct = width
                    h_correct = height

                # get sliding patch label
                img_label = [{'x':0, 'y':0, 'w':w_correct, 'h':h_correct, 'class':large_bbox_label['class']}]
                
                # save image and label to file
                fully_positive_patch_nums += 1
                cv2.imwrite(save_dir + 'train/' + img_name + '_{}_large.jpg'.format(fully_positive_patch_nums), img_arr)
                with open(save_dir + 'label/' + img_name + '_{}_large.json'.format(fully_positive_patch_nums), 'w') as json_f:
                    json.dump(img_label, json_f)
        return fully_positive_patch_nums

TRIAN_PATH = 'train'
SAVE_DIR = 'train/' # save generate data into SAVE_DIR
IMG_SIZE = 1000
STRIDE_PROPORTION = 0.5 # for sliding patch stride
CLASS_NAMES = ["ASC-H", "ASC-US", "HSIL", "LSIL", "Candida", "Trichomonas"]
json_paths  = glob.glob(TRIAN_PATH + '/*.json')

# fully positive patch image are produced under over-large bbox through sliding method
all_normal_patch_nums = 0 
all_fully_positive_patch_nums = 0

for p in json_paths:
    
    # load train json file to gain labels
    with open(p, 'r') as f:
        train_dicts = json.load(f)
    
    # read kfb image
    img_name = p.split('/')[-1].split('.')[0]
    kfb_path = os.path.join(TRIAN_PATH, img_name + '.kfb')
    read = kfbReader.reader()
    read.ReadInfo(kfb_path, 20, True)
    
    large_bbox_labels = []
    nums = 0
    for train_dict in train_dicts:
        if train_dict['class'] != 'roi':
            w = int(train_dict['w'])
            h = int(train_dict['h'])
            x_min = int(train_dict['x'])
            y_min = int(train_dict['y'])
            x_max = x_min + w - 1
            y_max = y_min + h - 1
            x_center = int(x_min + int(w/2) -1)
            y_center = int(y_min + int(h/2) -1 )
            
            if w * h < IMG_SIZE**2 and w < IMG_SIZE and h < IMG_SIZE:  
                # define the largest offset distance
                offset_dist= int((IMG_SIZE - max(w, h))/2)
    
                # the random center offset of the sliding patch
                rand_x_offset = np.random.randint(-offset_dist, offset_dist)
                rand_y_offset = np.random.randint(-offset_dist, offset_dist)

                # record new random patch information
                patch_label = {'x':x_center + rand_x_offset - IMG_SIZE//2,\
                                'y':y_center + rand_y_offset - IMG_SIZE//2,
                                'w':IMG_SIZE, 'h':IMG_SIZE}

                # obtain new bbox labels for a random patch
                new_bbox_labels = iou_filter(patch_label, train_dicts, 0.5)
                nums += 1

                # save patch into jpg image and its json label
                if not os.path.exists(SAVE_DIR):
                    os.makedirs(SAVE_DIR)
                    os.makedirs(SAVE_DIR + '/train/')
                    os.makedirs(SAVE_DIR + '/label/')

                img = read.ReadRoi(patch_label['x'], patch_label['y'], patch_label['w'], patch_label['h'], 20)
                cv2.imwrite(SAVE_DIR + 'train/' + img_name + '_{}.jpg'.format(nums), img)
                with open(SAVE_DIR + 'label/' + img_name + '_{}.json'.format(nums), 'w') as json_f:
                    json.dump(new_bbox_labels, json_f)
            else:
                large_bbox_labels.append(train_dict)
                
    if large_bbox_labels != []:
        # produce jpg and label for large bboxes
        fully_positive_patch_nums = produce_patch_for_large_bbox(large_bbox_labels, img_name, STRIDE_PROPORTION, IMG_SIZE, read, SAVE_DIR)
    else:
        fully_positive_patch_nums = 0

    all_normal_patch_nums += nums 
    all_fully_positive_patch_nums += fully_positive_patch_nums      
    print('Image-%s produces %d normal patch images and %d fully-positive patch image.' % (img_name, nums, fully_positive_patch_nums))
    
print('\nall normal image numbers: %d, all fully positive image numbers: %d' % (all_normal_patch_nums, all_fully_positive_patch_nums))
print('All patch images were saved to %s.' % SAVE_DIR)

# # visualize specific class image to check if the data generation is right
# fig = plt.figure(figsize=(16,16))
# p = 'train/label/1162_1_large.json'
# pp = 'train/train/1162_1_large.jpg'
# with open(p, 'r') as f:
#     train_dicts = json.load(f)
# img = cv2.imread(pp)


# for l in train_dicts:
#     x = l['x']
#     y = l['y']
#     w = l['w']
#     h = l['h']
#     img = cv2.rectangle(img,(x,y),(x+w-1,y+h-1),(0,255,0),4)
#     print(l['class'])

# plt.imshow(img)

# # compute the generated data distribution
# import glob
# json_paths = glob.glob('train/label/*.json')
# check_data_distribution(json_paths)


# # visualize the generated samples with different classes
# import glob
# import cv2
# import json

# json_paths = glob.glob('ahf_train/label/*.json')

# display_image_set = [] # store the samples

# for c in CLASS_NAMES:
#     break_flag = 0
#     for json_path in json_paths:       
#         # obtain image name 
#         img_name = json_path.split('/')[-1][:-5]
        
#         with open(json_path, 'r') as f:
#             labels = json.load(f)
        
#         for label in labels:
#             if label['class'] == c:
#                 x = int(label['x'])
#                 y = int(label['y'])
#                 w = int(label['w'])              
#                 h = int(label['h'])
#                 img = cv2.imread(SAVE_DIR + 'train/' + img_name + '.jpg')
#                 img = cv2.rectangle(img,(x,y), (x+w-1,y+h-1), (0,255,0), 4)
#                 display_image_set.append(img)
                
#                 break_flag = 1
#                 break
        
#         if break_flag == 1:
#             break

# # visualize the stored samples with different classes
# fig = plt.figure(figsize=(32,32))
# for i in range(len(CLASS_NAMES)):
#     plt.subplot(2,3,i+1)
#     plt.imshow(display_image_set[i])
#     plt.title(CLASS_NAMES[i])
#     plt.axis('off')
# plt.tight_layout() 
# plt.show()