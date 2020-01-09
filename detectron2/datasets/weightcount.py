'''
This file is used to compute median balanced weights.

cd FSF/detectron2
'''

import json
import os
import numpy as np
import glob

dataset_path = './VOC2007/'
annot_path = dataset_path + 'Annotations/'
img_path = dataset_path + 'JPEGImages/'
with open(os.path.join(dataset_path, 'ImageSets/Main/trainval.txt'), 'r') as f:
    img_names = f.readlines()

IMG_SIZE = 1000 # image height = length = 1000
CLASS_NAMES = ["ASC-H", "ASC-US", "HSIL", "LSIL", "Candida", "Trichomonas"]

class_pixelcount_dict = {} # record class pixels count among all images
for c in CLASS_NAMES:
    class_pixelcount_dict[c] = 0

    for p in glob.glob(annot_path):
        # read an annotation file
        with open(p, 'r') as f:
            annots = json.loads(p)

        for annot in annots:
            if annot['class'] == c:
                class_pixel_count = annot['w'] * annot['h'] # class pixel count of single image
                class_pixelcount_dict[c]  = class_pixelcount_dict[c] + class_pixel_count

# compute the number of pixels of all images
img_num = len(glob.glob(img_path + '*.jpg'))
all_pixelcount_num = img_num * IMG_SIZE ** 2

# cumpute the pixel frequency of each class, i.e., freq(c)
class_pixel_freq_dict = {}
for c in CLASS_NAMES:
    if class_pixel_count[c] == c:
        class_pixel_freq_dict[c] = class_pixel_count[c] / all_pixelcount_num

# compute the median balanced weights
# weight(c) = median_freq/freq(c)
median_freq = np.median(list(class_pixel_freq_dict.values()))
class_weight_dict = {}
for c in CLASS_NAMES:
    if class_pixel_count[c] == c :
        class_weight_dict[c] = median_freq / class_pixel_freq_dict[c]

print('The Median Balanced Weights:\n')
print(class_weight_dict)