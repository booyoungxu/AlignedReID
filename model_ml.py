import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from utils.resnet import resnet50
from utils.xception import xception


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class Model(nn.Module):
    def __init__(self, local_channels=128, num_classes=None):
        super(Model, self).__init__()
        self.base1 = resnet50(pretrained=True)
        self.base2 = xception(num_classes=num_classes)
        planes = 2048

        bn_relu = []
        bn_relu += [nn.BatchNorm2d(planes)]
        bn_relu += [nn.ReLU(inplace=True)]
        bn_relu = nn.Sequential(*bn_relu)
        bn_relu.apply(weights_init_kaiming)
        self.bn_relu = bn_relu

        local_layer = []
        local_layer += [nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 7), padding=0)]
        local_layer += [nn.Conv2d(planes, local_channels, 1)]
        local_layer = nn.Sequential(*local_layer)
        local_layer.apply(weights_init_kaiming)
        self.local_layer = local_layer

        global_layer = []
        global_layer += [nn.AvgPool2d(kernel_size=(7, 7), stride=(7, 7))]
        global_layer = nn.Sequential(*global_layer)
        self.global_layer = global_layer

        if num_classes is not None:
            classifier = []
            classifier += [nn.Linear(planes, num_classes)]
            classifier = nn.Sequential(*classifier)
            classifier.apply(weights_init_classifier)
            self.classifier = classifier

    def forward(self, x):
        feat1 = self.base1(x)
        feat1 = self.bn_relu(feat1)

        global_feat1 = self.global_layer(feat1)
        global_feat1 = global_feat1.view(global_feat1.size(0), -1)
        global_feat1 = global_feat1 * 1.0 / (torch.norm(global_feat1, 2, dim=-1, keepdim=True).expand_as(global_feat1) + 1e-8)

        local_feat1 = self.local_layer(feat1) # [N, 128, 7, 1]
        local_feat1 = local_feat1.squeeze(-1).permute(0, 2, 1) # [n, 7, 128]
        local_feat1 = local_feat1*1.0/(torch.norm(local_feat1, 2, dim=-1, keepdim=True).expand_as(local_feat1)+1e-8)

        feat2 = self.base2(x)
        feat2 = self.bn_relu(feat2)

        global_feat2 = self.global_layer(feat2)
        global_feat2 = global_feat2.view(global_feat2.size(0), -1)
        global_feat2 = global_feat2 * 1.0 / (torch.norm(global_feat2, 2, dim=-1, keepdim=True).expand_as(global_feat2) + 1e-8)

        local_feat2 = self.local_layer(feat2)  # [N, 128, 7, 1]
        local_feat2 = local_feat2.squeeze(-1).permute(0, 2, 1)  # [n, 7, 128]
        local_feat2 = local_feat2 * 1.0 / (torch.norm(local_feat2, 2, dim=-1, keepdim=True).expand_as(local_feat2) + 1e-8)

        if hasattr(self, 'classifier'):
            scores1 = self.classifier(global_feat1)
            scores2 = self.classifier(global_feat2)

            return global_feat1, local_feat1, scores1, global_feat2, local_feat2, scores2, global_feat1+global_feat2

        return global_feat1, local_feat1, global_feat2, local_feat2, global_feat1+global_feat2
