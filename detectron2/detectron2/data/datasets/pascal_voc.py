#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:32:53 2019

@author: alvinai
"""
from fvcore.common.file_io import PathManager
import os
import numpy as np
from PIL import Image
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


__all__ = ["register_pascal_voc"]


# fmt: off
CLASS_NAMES = [
    "ASC-H", "ASC-US", "HSIL", "LSIL", "Candida", "Trichomonas"
]

# fmt: on

# loads the dataset into detectron2's standard format
def load_voc_instances(dirname: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """    
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".json")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")
        
        im = Image.open(jpeg_file)
         
        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(im.size[1]),
            "width": int(im.size[0]),
        }
        instances = []
        
        with open(anno_file) as json_file:
            objs = json.load(json_file)
        
        for obj in objs:
            cls = obj['class']
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            
            # Make pixel indexes 0-based
            xmin = float(obj['x'])
            ymin = float(obj['y'])
            xmax = float(obj['x'] + obj['w'] - 1)
            ymax = float(obj['y'] + obj['h'] - 1)
            bbox = [xmin, ymin, xmax, ymax]

            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

def register_pascal_voc(name, dirname, split, year):
    DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, dirname=dirname, year=year, split=split
    )