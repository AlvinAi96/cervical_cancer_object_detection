#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is to split the dataset into Train and Validation dataset and produce
the corresponding txt file to list their file names.

author: Hongfeng Ai
date: 2019-12-5

cd detectron2/datasets/VOC2007
"""

from glob import glob
import random

# fetch all image file names together and shuffle them
patch_fn_list = glob('JPEGImages/*.jpg')
patch_fn_list = [fn.split('/')[-1][:-4] for fn in patch_fn_list]
random.shuffle(patch_fn_list)

# split the train/valid images with 7:3
train_num = int(0.7 * len(patch_fn_list))
train_patch_list = patch_fn_list[:train_num]
valid_patch_list = patch_fn_list[train_num:]

# produce train/valid/trainval txt file
split = ['train', 'val', 'trainval']

for s in split:
    save_path = 'ImageSets/Main/' + s + '.txt'

    if s == 'train':
        with open(save_path, 'w') as f:
            for fn in train_patch_list:
                f.write('%s\n' % fn)
    elif s == 'val':
        with open(save_path, 'w') as f:
            for fn in valid_patch_list:
                f.write('%s\n' % fn)
    elif s == 'trainval':
        with open(save_path, 'w') as f:
            for fn in patch_fn_list:
                f.write('%s\n' % fn)
    print('Finish Producing %s txt file to %s' % (s, save_path))