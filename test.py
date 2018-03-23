# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
# from model import Model
from model_ml import Model

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='210', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='/home/xufurong/data/images', type=str, help='./test_data')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')

test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

data_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.486, 0.459, 0.408], [0.229, 0.224, 0.225])
])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=4)
               for x in ['gallery', 'query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()


def load_network(network):
    save_path = os.path.join('./models', 'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, 2048).zero_()
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())
            global_feat1, local_feat1, scores1, global_feat2, local_feat2, scores2, combine_feat = model(input_img)
            # global_feat1, local_feat1, scores1 = model(input_img)
            f = combine_feat.data.cpu()
            ff = ff+f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

print('-------test-----------')
model_structure = Model(num_classes=751)
model = load_network(model_structure)

model.fc = nn.Sequential()

model = model.eval()
if use_gpu:
    model = model.cuda()
gallery_feature = extract_feature(model, dataloaders['gallery'])
query_feature = extract_feature(model, dataloaders['query'])

result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
scipy.io.savemat('test_1market1501.mat', result)

