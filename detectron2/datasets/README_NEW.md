### Folder: VOC2007
This folder has the following files:  
- Annotation: the label dataset for model. (e.g., 5634.json)  
- ImageSets/Main/: it contains train/trainval/val.txt which includes image names without suffix.  
- JPEGImages: the image dataset for model. (e.g., 5634.jpg)  
- test: the test dataset which involves kfb images and json labels for prediction.  

### split_dataset_produce_txt.py
This file can produce train/trainval/val.txt to VOC2007/ImageSets/Main/.  

### uniform_hue.py
This file is to uniform the hue value in VOC2007/JPEGImages.  

### weightcount.py
This file is used to compute median balanced weights which can be used on class-balanced loss.  