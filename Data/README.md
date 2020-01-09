### roi_based_data_generation.py  
This file is to generate data.  
The data generation uses roi-based sliding method.  

RUN: cd Data/roi_based_data_generation.py  

### pos_based_data_generation.py  
This file is to check data distribution and generate data.  
The data generation uses pos-center-based sliding method.  

RUN: cd Data/pos_based_data_generation.py  

### big_rotate_object.py
This file is to find out the patch image with large bbox, rotate them and then save them.   
The output result can be seen under the folder 'big_rotate_object'.  

RUN: cd Data/big_rotate_object.py  

### Folder: train
This folder contains original training dataset including KFB images and json labels.   

### Folder: test
This folder contains original test dataset including KFB images and json labels.   

### Folder: Train
This folder contains the following two folders produced by 'roi_based_data_generation.py'/'pos_based_data_generation.py':  
- train: the produced training jpg images.  
- label: the produced training json labels.  

### Folder: big_rotate_object
This folder contains rotated images with large bboxes and their labels, which produced by 'big_rotate_object.py'. More information, please visit 'big_rotate_object/README.md'.  