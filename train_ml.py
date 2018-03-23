# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F

from PIL import Image
import time
import os
import sys
import json
import argparse
from collections import defaultdict

from model_ml import Model
from utils.loss import triplet_loss, TripletLoss, mixed_loss
from utils.utils import Logger


class TripletSampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.pids = defaultdict(list)
        for idx in np.arange(len(data_source)):
            self.pids[data_source[idx][1]].append((data_source[idx][0], data_source[idx][1], idx))
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples*4 * self.num_samples

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        labels = []
        for i in indices:
            if len(self.pids[i]) >= self.num_instances:
                tmp = np.random.choice(np.arange(len(self.pids[i])), size=self.num_instances, replace=False)
            else:
                tmp = np.random.choice(np.arange(len(self.pids[i])), size=self.num_instances, replace=True)
            for j in tmp:
                image, target, idx = self.pids[i][j]
                ret.append(idx)
                labels.append(target)
        return iter(ret)


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--data_dir', default='/home/xufurong/data/images', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--num_instances', default=4, type=int, help='the number of instances for every ID')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')

opt = parser.parse_args()

data_dir = opt.data_dir
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)


if len(gpu_ids) > 0:
    # os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    torch.cuda.set_device(gpu_ids[0])

transform_train_list = [
        transforms.Resize((260, 260), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.486, 0.459, 0.408], [0.229, 0.224, 0.225])
        ]


transform_val_list = [
        transforms.Resize(size=(224, 224),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.486, 0.459, 0.408], [0.229, 0.224, 0.225])
        ]


data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all), data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, num_workers=4, drop_last=True,
                                              sampler=TripletSampler(image_datasets[x], opt.num_instances))
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))


def adjust_lr(optimizer, epoch):
    lr = opt.lr if epoch <= 76 else \
        opt.lr * (0.001 ** ((epoch - 76) / 38))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)


def train_model(model, cls_criterion, t_criterion, optimizer, num_epochs=25):
    since = time.time()
    last_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        adjust_lr(optimizer, epoch)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            lt1 = 0.0
            lm1 = 0.0
            lt2 = 0.0
            lm2 = 0.0
            cnt = 0
            for data in dataloaders[phase]:
                inputs, labels = data
                cnt += len(labels)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                global_feat1, local_feat1, output1, global_feat2, local_feat2, output2, combine_feat = model(inputs)
                prob1 = F.softmax(output1, dim=1)
                prob2 = F.softmax(output2, dim=1)
                log_prob1 = F.log_softmax(output1, dim=1)
                log_prob2 = F.log_softmax(output2, dim=1)
                loss_cls1 = cls_criterion(output1, labels)
                loss_cls2 = cls_criterion(output2, labels)
                loss_tri1, global_dist_mat1 = mixed_loss(t_criterion, global_feat1, local_feat1, labels, mutual_feature=True)
                loss_tri2, global_dist_mat2 = mixed_loss(t_criterion, global_feat2, local_feat2, labels, mutual_feature=True)
                loss_mutual_cls1 = F.kl_div(log_prob1, prob2.detach(), False)
                loss_mutual_cls2 = F.kl_div(log_prob2, prob1.detach(), False)
                # loss_mutual_cls2 = (output2.detach()*torch.log(output2.detach()/(output1+1e-8)+1e-8)).sum()
                loss_mutual_tri1 = ((global_dist_mat1.detach()-global_dist_mat2)**2).mean(0).mean(0)
                loss_mutual_tri2 = ((global_dist_mat2.detach()-global_dist_mat1)**2).mean(0).mean(0)
                # loss = loss_cls1+loss_cls2+loss_tri1+loss_tri2+loss_mutual_tri1+loss_mutual_tri2+0.01*(loss_mutual_cls1+loss_mutual_cls2)
                loss = loss_tri1+loss_tri2+loss_mutual_tri1+loss_mutual_tri2
                # loss_tri_all = mixed_loss(t_criterion, global_feat, local_feat, labels, mutual_feature=False)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                lt1 += loss_tri1.data[0]
                lm1 += loss_mutual_tri1.data[0]
                lt2 += loss_tri2.data[0]
                lm2 += loss_mutual_tri2.data[0]

            epoch_loss = running_loss / cnt

            print('{} Loss: {:.4f} lt1: {:.4f} lm1: {:.4f} lt2: {:.4f} lm2: {:.4f}'.format(phase, epoch_loss, lt1, lm1, lt2, lm2))

            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch+1)
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    model.load_state_dict(last_model_wts)
    save_network(model, 'last')

    return model


def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./models0', save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])


model = Model(num_classes=len(class_names))

if use_gpu:
    model = model.cuda()

id_criterion = nn.CrossEntropyLoss()
tri_criterion = TripletLoss(margin=0.6)


optimizer_ft = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)


dir_name = os.path.join('./models0', '')
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)

with open('%s/opts.json' % dir_name, 'w') as fp:
    json.dump(vars(opt), fp, indent=1)

sys.stdout = Logger(os.path.join(dir_name, 'log.txt'))

model = train_model(model, id_criterion, tri_criterion, optimizer_ft, num_epochs=300)

