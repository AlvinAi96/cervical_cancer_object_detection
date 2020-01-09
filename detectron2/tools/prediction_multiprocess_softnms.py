#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:40:18 2019

@author: alvinai

This file is to predict test dataset and format the final result.
MODIFICATION: NMS threshod, class confidence threshold, NMS locations differs for classes

cd FSF/detectron2
"""

import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch
from detectron2.data import MetadataCatalog
import kfbReader
import copy
from collections import OrderedDict
import json
import time
import glob   
import numpy as np
import argparse
import multiprocessing
from cpu_soft_nms import cpu_soft_nms

def ReadInfo(self, kfbPath, scale=0, readAll=False):
    '''
    FUNCTION:
        kfbreader setting
    
    
    INPUTS::
        1. kfbPath: kfb document path
        2. scale:   scaling size
        3. readAll: whether read all information or not
    '''
    return kfbReader.reader.ReadInfo(self, kfbPath, scale, readAll)


def py_cpu_nms(dets, thresh):
    '''
    FUNCTION:
        NMS implementation
        
    
    INPUTS:
        1. dets (list): a list of detection information. 
                        format: [[xmin,ymin,xmax,ymax,p], [], ...]
        2. thresh (float): NMS threshold. only leave the detection with IOU < thresh
    
    
    OUTPUTS:
        1. keep (list): a list of indexes to keep
    '''
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]

    areas = (y2-y1+1) * (x2-x1+1)
    scores = dets[:,4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size >0:
        i = index[0]       # every time the first is the biggst, and add it directly
        keep.append(i)
 
 
        x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        
 
        w = np.maximum(0, x22-x11+1)    # the width of overlap
        h = np.maximum(0, y22-y11+1)    # the height of overlap
       
        overlaps = w*h
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
 
        idx = np.where(ious<=thresh)[0]
        index = index[idx+1]   # because index start from 1
 
    return keep


def prediction_single_roi(kfb_img_path, roi_dict, img_size, stride, kfb_scale,
                       predictor, output_save_path, class_names, FINAL_NMS_SWITCH,
                        FINAL_NMS_THRESH_DICT,CONF_THRESH_DICT):
    '''
    FUNCTION: 
        Predict all 1000x1000 small image splitted from a ROI.
        The input ROI has one of the following three types:
            1. large: ROI_w and ROI_h are all larger than 1000.
            2. medium: only ROI_w or ROI_h is larger than 1000.
            3. small: ROI_w and ROI_h are all smaller than 1000.
    
    INPUTS:
        1. kfb_image_path (str): a path to an image kfb file 
        2. roi_dict (dict): a ROI dict. e.g., {"x": 4670, "y": 20127, "w": 3189, "h": 3174, "class": "roi"}
        3. img_size (int): image size in training process. i.e.,1000
        4. stride (int): the sliding stride. e.g., 1000*(1/2)
        5. kfb_scale (int): the scale of kfb file. e.g., 20
        6. predictor: DefaultPredictor used to get prediction result
        7. output_save_path (str): the output saving path
        8. class_names (list): a list of class names
        9. FINAL_NMS_SWITCH (bool)： whether to do NMS
        4. FINAL_NMS_THRESH_DICT (dict): a dict of different NMS thresholds for different classes
        5. CONF_THRESH_DICT (dict): a dict of different confidence thresholds for differents classes
    
    OUTPUTS:
        1. prediction (list): the prediction result. the list is formated as follows:
            [{"x": 22890, "y": 3877, "w": 96, "h": 55，"p": 0.94135，"class": "ASC-H"}, 
             {"x": 20411, "y": 2260, "w":25, "h": 83，"p": 0.67213，"class": "ASC-US"}, 
             {"x": 26583, "y": 7937, "w": 72, "h": 128，"p": 0.73228，"class": "Candida"}]
    '''
    time_start = time.time()
    
    im_name = kfb_img_path.split('/')[-1][:-4]
    
    # read kfb file
    read = kfbReader.reader()
    read.ReadInfo(kfb_img_path, kfb_scale, False)
    
    # get the height/width/scale of the ROI image
    H = roi_dict['h']
    W = roi_dict['w']
    S = kfb_scale
    
    # get the horizontal/vertical moving step (i.e., sliding step)
    HOR_MOV_STEP = int((W - img_size)/stride + 1)
    VER_MOV_STEP = int((H - img_size)/stride + 1)    
     
    x = roi_dict['x']
    y = roi_dict['y']
    w = copy.deepcopy(img_size)
    h = copy.deepcopy(img_size)
    
    # get the total number of 1000x1000 images from an kfb image file
    total_img_arr_num = VER_MOV_STEP * HOR_MOV_STEP 
    
    prediction = []
    
    # VER_MOV_STEP != 0 and HOR_MOV_STEP != 0 means sliding patch is far smaller than roi
    if VER_MOV_STEP != 0 and HOR_MOV_STEP != 0:
        roi_type = 'large'
        # two forloops for horizontal and vertical moving the windows
        for ver in range(VER_MOV_STEP):
            for hor in range(HOR_MOV_STEP):
                x_new = x + hor * stride
                y_new = y + ver * stride
                
                # get a small image array (1000*1000)
                img_arr = read.ReadRoi(x_new, y_new, w, h, S)
                
                # get prediciton result
                output = predictor(img_arr)
                instances = output["instances"].to(torch.device("cpu"))
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.tolist()
                classes = instances.pred_classes.tolist()            
                
                for box, score, cls in zip(boxes, scores, classes):
                    if  score < CONF_THRESH_DICT[class_names[cls]]:
                        continue
                    else: 
                        xmin, ymin, xmax, ymax = box
                        width  = xmax - xmin + 1
                        height =  ymax - ymin + 1                
                        cls_name = class_names[cls]  
                                    
                        
                        # format prediction reuslt
                        pred_result = OrderedDict()
                        pred_result['x'] = int(xmin + x_new)
                        pred_result['y'] = int(ymin + y_new)
                        pred_result['w'] = int(width)
                        pred_result['h'] = int(height)
                        pred_result['p'] = float(round(score, 5))
                        pred_result['class'] = str(cls_name)
                        
                        # format: [{'x','y','w','h','p','class'}, {}, ...]
                        prediction.append(pred_result) 
    
    # total_img_arr_num = 0 means roi_h or roi_w is smaller than IMG_SIZE，which cannot be slided 
    elif VER_MOV_STEP == 0 and HOR_MOV_STEP == 0:
        roi_type = 'small'
        img_arr = read.ReadRoi(x, y, w, h, S)

        # get prediciton result
        output = predictor(img_arr)
        instances = output["instances"].to(torch.device("cpu"))
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        
        for box, score, cls in zip(boxes, scores, classes):
            if  score < CONF_THRESH_DICT[class_names[cls]]:
                continue
            else: 
                # roi-based coordinate
                xmin, ymin, xmax, ymax = box
                # convert to kfb-based coordinate
                bbox_xmin = int(xmin + x)
                bbox_ymin = int(ymin + y)
                bbox_xmax = int(xmax + x)
                bbox_ymax = int(ymax + y)

                # if the bbox outside the original roi
                if (bbox_xmin > x + W -1) or (bbox_ymin > y + H -1):
                    continue
                else:
                    new_bbox_xmin = int(max(bbox_xmin, x))
                    new_bbox_ymin = int(max(bbox_ymin, y))
                    new_bbox_xmax = int(min(bbox_xmax, x+W-1))
                    new_bbox_ymax = int(min(bbox_ymax, y+H-1))
                    new_bbox_w = int(max(0, new_bbox_xmax-new_bbox_xmin+1))
                    new_bbox_h = int(max(0, new_bbox_ymax-new_bbox_ymin+1))
                    if new_bbox_w * new_bbox_h > 0:
                        pred_result = OrderedDict()
                        pred_result['x'] = new_bbox_xmin
                        pred_result['y'] = new_bbox_ymin
                        pred_result['w'] = new_bbox_w
                        pred_result['h'] = new_bbox_h
                        pred_result['p'] = float(round(score, 5))
                        pred_result['class'] = str(class_names[cls])
                        
                        # format: [{'x','y','w','h','p','class'}, {}, ...]
                        prediction.append(pred_result)                    

    # VER_MOV_STEP == 0 or HOR_MOV_STEP == 0 means patch can only slide along one of direction   
    else:
        roi_type = 'medium'
        if VER_MOV_STEP == 0:
            W = img_size      
            ver = 0
            for hor in range(HOR_MOV_STEP):
                x_new = x + hor * stride
                y_new = y + ver * stride
                
                # get a small image array (1000*1000)
                img_arr = read.ReadRoi(x_new, y_new, w, h, S)
                
                # get prediciton result
                output = predictor(img_arr)
                instances = output["instances"].to(torch.device("cpu"))
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.tolist()
                classes = instances.pred_classes.tolist()

                for box, score, cls in zip(boxes, scores, classes):
                    if  score < CONF_THRESH_DICT[class_names[cls]]:
                        continue
                    else: 
                        # roi-based coordinate
                        xmin, ymin, xmax, ymax = box
                        # convert to kfb-based coordinate
                        bbox_xmin = int(xmin + x_new)
                        bbox_ymin = int(ymin + y_new)
                        bbox_xmax = int(xmax + x_new)
                        bbox_ymax = int(ymax + y_new)
                        
                        # if the bbox outside the original roi
                        if (bbox_xmin > x_new + W -1) or (bbox_ymin > y_new + H -1):
                            pass
                        else:
                            new_bbox_xmin = int(max(bbox_xmin, x_new))
                            new_bbox_ymin = int(max(bbox_ymin, y_new))
                            new_bbox_xmax = int(min(bbox_xmax, x_new+W-1))
                            new_bbox_ymax = int(min(bbox_ymax, y_new+H-1))
                            new_bbox_w = int(max(0, new_bbox_xmax-new_bbox_xmin+1))
                            new_bbox_h = int(max(0, new_bbox_ymax-new_bbox_ymin+1))
                            if new_bbox_w * new_bbox_h > 0:

                                pred_result = OrderedDict()
                                pred_result['x'] = new_bbox_xmin
                                pred_result['y'] = new_bbox_ymin
                                pred_result['w'] = new_bbox_w
                                pred_result['h'] = new_bbox_h
                                pred_result['p'] = float(round(score, 5))
                                pred_result['class'] = str(class_names[cls])
                                
                                # format: [{'x','y','w','h','p','class'}, {}, ...]
                                prediction.append(pred_result)  
        else:
            H = img_size   
            hor = 0
            for ver in range(VER_MOV_STEP):
                x_new = x + hor * stride
                y_new = y + ver * stride
                
                # get a small image array (1000*1000)
                img_arr = read.ReadRoi(x_new, y_new, w, h, S)
                
                # get prediciton result
                output = predictor(img_arr)
                instances = output["instances"].to(torch.device("cpu"))
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.tolist()
                classes = instances.pred_classes.tolist()

                for box, score, cls in zip(boxes, scores, classes):
                    if  score < CONF_THRESH_DICT[class_names[cls]]:
                        continue
                    else: 
                        # roi-based coordinate
                        xmin, ymin, xmax, ymax = box
                        # convert to kfb-based coordinate
                        bbox_xmin = int(xmin + x_new)
                        bbox_ymin = int(ymin + y_new)
                        bbox_xmax = int(xmax + x_new)
                        bbox_ymax = int(ymax + y_new)
                        
                        # if the bbox outside the original roi
                        if (bbox_xmin > x_new + W -1) or (bbox_ymin > y_new + H -1):
                            pass
                        else:
                            new_bbox_xmin = int(max(bbox_xmin, x_new))
                            new_bbox_ymin = int(max(bbox_ymin, y_new))
                            new_bbox_xmax = int(min(bbox_xmax, x_new+W-1))
                            new_bbox_ymax = int(min(bbox_ymax, y_new+H-1))
                            new_bbox_w = int(max(0, new_bbox_xmax-new_bbox_xmin+1))
                            new_bbox_h = int(max(0, new_bbox_ymax-new_bbox_ymin+1))
                            if new_bbox_w * new_bbox_h > 0:
                                pred_result = OrderedDict()
                                pred_result['x'] = new_bbox_xmin
                                pred_result['y'] = new_bbox_ymin
                                pred_result['w'] = new_bbox_w
                                pred_result['h'] = new_bbox_h
                                pred_result['p'] = float(round(score, 5))
                                pred_result['class'] = str(class_names[cls])
                                
                                # format: [{'x','y','w','h','p','class'}, {}, ...]
                                prediction.append(pred_result)  
    
    before_nms_dets_num = len(prediction)

    if FINAL_NMS_SWITCH == True:
        pred = copy.deepcopy(prediction)
        prediction = []
        for cls_ind, cls_n in enumerate(class_names):
            dets2 = [] # format: [[xmin,ymin,xmax,ymax,p], [], ...]
            PEDS = [] # format: [{'x','y','w','h','p','class'}, {}, ...]
            for p in pred:                
                if p['class'] == cls_n:
                    det = [p['x'], p['y'],
                           p['x']+p['w']-1, p['y']+p['h']-1,
                           p['p']]
                    dets2.append(det)
                    PEDS.append(p)
                       
            # if certain class has no detection, jump to next forloop       
            if np.array(dets2).shape[0] == 0:
                continue
            else:
                nms_threshold = FINAL_NMS_THRESH_DICT[cls_n]
                # SOFT NMS
                keep2 = cpu_soft_nms(boxes=np.array(dets2), sigma=0.5, Nt=0.1, threshold=0.05, method=1)
                PEDS = [PEDS[i] for i in keep2]
                prediction.extend(PEDS)

    after_nms_dets_num = len(prediction)
    time_end = time.time()

    print('Image %s-%s_roi %s | total_img_arr_num: %d | before_nms_dets_num:  %d | after_nms_dets_num:  %d | time: %.3f' \
          % (im_name, roi_type, str(roi_dict), total_img_arr_num, before_nms_dets_num, after_nms_dets_num, time_end-time_start))

    return prediction


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN prediction')
    parser.add_argument('--output_config', dest='output_config',
                        help='The path to the output config .yaml file',
                        default="./output/config.yaml")
    parser.add_argument('--model_weights_pth', dest='model_weights_pth',
                        help='The final model weights .pth file',
                        default='model_final.pth')
    parser.add_argument('--conf_thresh', dest='conf_threshold',
                        help='cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST',
                        default=0.05, type=float)
    parser.add_argument('--final_nms_switch', dest='final_nms_switch',
                        help='Final NMS switch',
                        default=True, type=bool)
    parser.add_argument('--img_size', dest='img_size', 
                        help='The image length/height', default=1000, type=int)
    parser.add_argument('--stride_proportion', dest='stride_proportion',
                        help='The proportion of stride', default=0.25, type=float)    
    parser.add_argument('--save_path', dest='save_path',
                        help='The path to save predicted results',
                        default='./output/submit_result/')
    parser.add_argument('--test_kfb_path', dest='test_kfb_path',
                        help='The path to test kfb files',
                        default='./datasets/VOC2007/test/')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    # set congfiguration
    cfg = get_cfg()
    cfg.merge_from_file(args.output_config)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_weights_pth)
    
    
    
    # obtain class_names info
    meta = MetadataCatalog.get('voc_2007_trainval')
    class_names = meta.thing_classes
    
    # Predictor conf/nms threshold setting. Personal suggestion: don't change them
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_threshold # CONF_THRESHOLD
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5 # NMS_THRESHOLD
    print('Configuration:\n', cfg)
    
    
    IMG_SIZE = args.img_size
    STRIDE = int(IMG_SIZE * args.stride_proportion)
    # After collecting all predicted boxes from many sliding small images,
    # they need NMS to filter tedious boxes due to the overlapping sliding method.
    # only when IOU<FINAL_NMS_THRESH_DICT['CLASS'], the box can be saved. It is available to change it.
    FINAL_NMS_SWITCH = args.final_nms_switch # whether to do NMS on all final detections
    #FINAL_NMS_THRESH_DICT = {"ASC-H":0.5, "ASC-US":0.4, "HSIL":0.1, "LSIL":0.1, "Candida":0.1, "Trichomonas":0.4}
    FINAL_NMS_THRESH_DICT = {"ASC-H":0.1, "ASC-US":0.1, "HSIL":0.1, "LSIL":0.1, "Candida":0.1, "Trichomonas":0.1}
    print('FINAL_NMS_SWITCH: ', FINAL_NMS_SWITCH)
    print('FINAL_NMS_THRESH_DICT:', FINAL_NMS_THRESH_DICT)
    
    # only when p>CONF_THRESH['CLASS'], the box can be saved. It is available to change it.
    CONF_THRESH_DICT = {"ASC-H":0.05, "ASC-US":0.05, "HSIL":0.05, "LSIL":0.05, "Candida":0.05, "Trichomonas":0.05}
    print('CONF_THRESH_DICT:', CONF_THRESH_DICT)

    # global NMS without the restrition of class, which ensure there is not overlapping boxes among multiple classes
    # 1: do global NMS
    #GLOBAL_NMS_LIST = {"ASC-H":0, "ASC-US":0, "HSIL":0, "LSIL":0, "Candida":1, "Trichomonas":1}
    GLOBAL_NMS_LIST = {"ASC-H":0, "ASC-US":0, "HSIL":0, "LSIL":0, "Candida":0, "Trichomonas":0}
    
    output_save_path = args.save_path + '_' + str(cfg.SOLVER.MAX_ITER)
    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path)
        
    test_kfb_path = glob.glob(args.test_kfb_path + '*.kfb')

    # sort test_kfb_path by roi number with descending order
    roi_num_list = []
    for kfb_path in test_kfb_path:
        with open(kfb_path[:-4] + '.json', 'r') as f:
            labels = json.load(f)
        roi_num_list.append(len(labels))
    
    test_kfb_path_sorted = []
    roi_num_unique_list = sorted(np.unique(roi_num_list), reverse=True)
    for roi_num in roi_num_unique_list:
        for i in range(len(roi_num_list)):
            if roi_num_list[i] == roi_num:
                test_kfb_path_sorted.append(test_kfb_path[i])                

    def mp_func(path, gpu_id):
        '''Multiprocessing Function'''
        torch.cuda.set_device(gpu_id)
        
        predictor = DefaultPredictor(cfg)
        
        # get roi labels for test json file
        image_name = path.split('/')[-1][:-4]
        with open(args.test_kfb_path + str(image_name) + '.json', 'r') as json_f:
            roi_labels = json.load(json_f)
            
        # prediction all rois under certain KFB image file
        PREDICTIONS = []
        for r in roi_labels:
            result = prediction_single_roi(path, r, IMG_SIZE, STRIDE, 20,
                                            predictor, output_save_path,
                                            class_names, FINAL_NMS_SWITCH,
                                            FINAL_NMS_THRESH_DICT, CONF_THRESH_DICT)
            PREDICTIONS.extend(result)
              
        # save prediction result to json file
        with open(output_save_path + str(image_name) + '.json', 'w') as f:
            json.dump(PREDICTIONS, f)
        print('\n')
    
    TIME_START = time.time()   
    print('Start Multiprocessing:') 
       
    img_num_count = 0
    process_num = 2 # suggest it equal to gpu_num  
    for process_count in range(int(len(test_kfb_path_sorted)/process_num)):
        start_ind = process_count * process_num
        end_ind = process_count * process_num + process_num
             
        processes = []
        for gpu_id, kfb_img_path in enumerate(test_kfb_path_sorted[start_ind:end_ind]):            
            img_num_count += 1
            print('Prediciton Process: %d/%d' % (img_num_count, len(test_kfb_path_sorted)))
            
            pro = multiprocessing.Process(target=mp_func, args=(kfb_img_path, gpu_id))
            processes.append(pro)   
        
        for j in range(process_num):
            processes[j].daemon = False
            processes[j].start()
        
        for j in range(process_num):
            processes[j].join()
            time.sleep(2)

    print('All predicted results were outputed to %s' % output_save_path)
    
    TIME_END = time.time()
    time_cost = (TIME_END-TIME_START)/60
    print('Prediction takes %.3f minutes' % time_cost)