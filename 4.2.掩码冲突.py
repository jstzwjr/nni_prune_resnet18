# 当对具有通道依赖关系的层设置不同的稀疏度，可以通过maskconflict来修复


import torch
import torchvision
from torchvision.models import resnet18
import torch.nn as nn
from nni.algorithms.compression.pytorch.pruning import FPGMPruner
from cifar_resnet import ResNet18
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import logging
from nni.compression.pytorch.speedup import ModelSpeedup
from nni.compression.pytorch import apply_compression_results
from nni.compression.pytorch.utils.counter import count_flops_params
from copy import deepcopy
from nni.compression.pytorch.utils.mask_conflict import fix_mask_conflict


device = "cuda:0"
dummy_input = torch.rand((1000,3,32,32)).to(device)


model = ResNet18(10)
ori_state_dict = torch.load("checkpoint_best.pt", map_location="cuda")
model.load_state_dict(ori_state_dict)
model = model.cuda().eval()

mask_path = 'mask.pth'
fixed_mask_path = "fixed_mask.pth"

# 掩码冲突校验，如果对有通道依赖关系的层进行剪枝，需只修剪公共的部分
new_mask = fix_mask_conflict(mask_path, model, dummy_input)
torch.save(new_mask, fixed_mask_path)


# 对比fix前后mask的差别
old_mask = torch.load(mask_path)
# for name, val in old_mask.items():
#     weight = old_mask[name]["weight"]
#     weight = weight.abs().sum(dim=(1,2,3))
#     print((weight==0).sum())

new_mask = torch.load(fixed_mask_path)
# for name, val in old_mask.items():
#     weight = new_mask[name]["weight"]
#     weight = weight.abs().sum(dim=(1,2,3))
#     print((weight==0).sum())


for name, val in old_mask.items():
    old_weight = old_mask[name]["weight"]

    new_weight = new_mask[name]["weight"]

    fixed_weight = old_weight + new_weight
    fixed_weight = fixed_weight.abs().sum(dim=(1,2,3))
    print((fixed_weight==0).sum())

