import os
from shutil import copyfile

data_root = '/home/xufurong/data'
download_path = '/home/xufurong/data/market1501/Market-1501-v15.09.15'

if not os.path.isdir(download_path):
    print('please ensure the download_path')

save_path = os.path.join(data_root, 'images')

if not os.path.isdir(save_path):
    os.makedirs(save_path)

query_path = os.path.join(download_path, 'query')
query_save_path = os.path.join(save_path, 'query')
if not os.path.isdir(query_save_path):
    os.makedirs(query_save_path)

for _, _, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = os.path.join(query_path, name)
        dst_path = os.path.join(query_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)
        copyfile(src_path, os.path.join(dst_path, name))

gallery_path = os.path.join(download_path, 'bounding_box_test')
gallery_save_path = os.path.join(save_path, 'gallery')
if not os.path.isdir(gallery_save_path):
    os.makedirs(gallery_save_path)

for _, _, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = os.path.join(gallery_path, name)
        dst_path = os.path.join(gallery_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)
        copyfile(src_path, os.path.join(dst_path, name))


train_all_path = os.path.join(download_path, 'bounding_box_train')
train_all_save_path = os.path.join(save_path, 'train_all')
if not os.path.isdir(train_all_save_path):
    os.makedirs(train_all_save_path)

for _, _, files in os.walk(train_all_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = os.path.join(train_all_path, name)
        dst_path = os.path.join(train_all_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)
        copyfile(src_path, os.path.join(dst_path, name))


train_save_path = os.path.join(save_path, 'train')
val_save_path = os.path.join(save_path, 'val')
if not os.path.isdir(train_save_path):
    os.makedirs(train_save_path)
    os.makedirs(val_save_path)

for _, _, files in os.walk(train_all_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = os.path.join(train_all_path, name)
        dst_path = os.path.join(train_save_path, ID[0])
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)
            dst_path = os.path.join(val_save_path, ID[0])  #first image is used as val image
            os.makedirs(dst_path)
        copyfile(src_path, os.path.join(dst_path, name))
