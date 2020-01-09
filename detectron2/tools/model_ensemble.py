'''
This file is to implemtent Model Ensemble based on different performance of different experiments.

author: Hongfeng Ai
date: 2020-01-08
'''
import glob
import json
import os
import time


# define which class results you want in different experiments
iter_class_dict = {'output/submit_result_AH':["ASC-H"],
                   'output/submit_result_AS':["ASC-US"],
                   'output/submit_result_HL_LL':["HSIL", "LSIL"],
                   'output/submit_result_CA':["Candida"],
                   'output/submit_result_TS':["Trichomonas"]}

# get image names of test dataset
test_names =  [str(p.split('/')[-1][:-5]) for p in glob.glob('output/submit_result_AH/*.json')]

# save path
save_path = 'output/ensemble_submit_result'
if not os.path.exists(save_path):
    os.makedirs(save_path)

print('Start saving ensemble result:')
TIME_START = time.time()

img_count = 0
for test_name in test_names:
    img_count += 1
    prediction = []
    # extract target-class result iteration by iteration
    for p in list(iter_class_dict.keys()):
        json_path = p + '/' + test_name + '.json' 
        with open(json_path, 'r') as json_file:
            roi_dicts = json.load(json_file)
            for roi_dict in roi_dicts:
                if roi_dict['class'] in iter_class_dict[p]:
                    prediction.append(roi_dict)
    
    # save ensemble result
    with open(save_path + '/' + test_name + '.json', 'w') as save_f:
        json.dump(prediction, save_f)

    print('%d/%d: %s' % (img_count, len(test_names), test_name))

TIME_END = time.time()
time_cost = (TIME_END-TIME_START)/60
print('Finish saving ensemble result to %s （it takes %.3f minutes）' % (save_path, time_cost))