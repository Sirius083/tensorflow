# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 10:38:06 2019

@author: Sirius

pre-trained resnet 
"""


# Note
# panda_169: crane is duplicate (both 545 and 429)
# panda_389: crane is duplicate (both 518 and 135)

# validation picture directories
# move wrong prediction to directory
import os
from shutil import copyfile
from imagenet_validation_test import predict
from imagenet_preprocessing import preprocess_image

# Note: 列一个总目录，则可以尽可能少改下面的子路径，多用os.path.join
work_dir = r'E:\resnet_test\predict' 
data_dir = r'E:\ImageNet2012'
wrong_dir = os.path.join(data_dir, 'img_val_wrong')  # directory for wrong images

# define file path and directory
vali_dir = os.path.join(data_dir, 'ILSVRC2012_img_val')
path_uid_to_labels = os.path.join(work_dir, 'imagenet_2012_validation_synset_labels.txt') # validation_uid
path_uid_to_name = os.path.join(work_dir, 'imagenet_synset_to_human_label_map.txt')
path_cls_to_name_panda_389 = os.path.join(work_dir, 'imagenet1000_clsidx_to_labels.txt')

# process panda_389
with open(path_cls_to_name_panda_389, 'r') as f:
	panda_389 = f.readlines()

cls_to_name_389 = {}
for x in panda_389:
    k,v = x.rstrip().split(':')
    cls_to_name_389[int(k)+1] = v[2:-2]
    
# read uid_to_labels (true label)
with open(path_uid_to_labels,'r') as f:
	content = f.readlines()
content = [c.rstrip() for c in content]

# read uid_to_name mapping
with open(path_uid_to_name, 'r') as f:
    uid_to_name_mapping = f.readlines()
uid_to_name_mapping = [t.rstrip() for t in uid_to_name_mapping]
uid_to_name = {}

for x in uid_to_name_mapping:
    k,v = x.split('\t')
    uid_to_name[k] = v

#=====================================
#         predict label
#=====================================
path_list = os.listdir(vali_dir)
# tmp_name = 'ILSVRC2012_val_00000210.JPEG'
# path_list = path_list[path_list.index(tmp_name):]

wrong_list = []
for pic_img_name in path_list:
    # img_path = r'E:\ImageNet2012\ILSVRC2012_img_val\ILSVRC2012_val_00000001.jpeg'

    print('===============')
    print(pic_img_name)
    img_path = os.path.join(vali_dir, pic_img_name)
    top10_id, top10_prob = predict(img_path)
    predict_name = cls_to_name_389[top10_id[0]]
    
    #======================================
    #         correct label
    #====================================== 
    img_num = int(img_path.split('.')[0].split('_')[-1]) # image number in validation set
    img_uid = content[img_num-1]    # corresponding syntex number
    img_name = uid_to_name[img_uid] # correct name
    
    if img_name != predict_name:
       print('image name: ',img_name)
       print('predi name:' ,predict_name)
       # 标出在验证集的标号：否则重复认错误
       ori_name = os.path.join(wrong_dir, pic_img_name)
       copyfile(img_path, ori_name) # copy file to another directory
       new_name = os.path.join(wrong_dir, img_name.split(',')[0] + '__' + predict_name.split(',')[0])
       new_name =  new_name + '_' + str(img_num) + '.jpeg'
       os.rename(ori_name,new_name)
       wrong_list = wrong_list + img_num
       
    
    
    
    
    
    
    
    









