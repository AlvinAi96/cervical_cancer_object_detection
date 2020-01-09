'''
cd FSF/detectron2/datasets
'''
import os
import cv2
import glob
import numpy as np

def uniform_hue(img_path, hv, img_size, save_path):
    '''
    Function:
        change all BRG image with the same hue
    
    Args:
        1. img_path (str): the path to jpg image dataset
        2. hv (int): the specific hue value we want to change
        3. img_size (int): the image size (default 1000)
        4. save_path (str): the saving path of images with uniform hue.

    Return:
        save converted images
    ''' 
    img_paths = glob.glob(img_path + '*.jpg')
    count = 0 
    for p in img_paths:
        img_name = p.split('/')[-1][:-4]

        # read BGR image
        img = cv2.imread(p)
        
        # convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # create a new uniform hue array
        h_new = np.full((img_size, img_size), hv).astype('uint8')
        img_new = cv2.merge((h_new, hsv[:,:,1], hsv[:,:,2]))
        
        # hsv to bgr
        img_new_bgr = cv2.cvtColor(img_new, cv2.COLOR_HSV2BGR)

        # save image with uniform hue
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        count+=1
        cv2.imwrite(save_path + '{}.jpg'.format(img_name), img_new_bgr)
        print('%d/%d: %s' % (count, len(img_paths), img_name))


img_path = './VOC2007/JPEGImages/'
save_path = './VOC2007/JPEGImages_uniform_hue/'

uniform_hue(img_path, 100, 1000, save_path)