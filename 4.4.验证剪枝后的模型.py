import torch
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
from nni.compression.pytorch.utils.mask_conflict import fix_mask_conflict

logger = logging.getLogger(__name__)

device = "cuda:0"
dummy_input = torch.rand((1000, 3, 32, 32)).to(device)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]), download=True), batch_size=512, num_workers=16)


def test(model):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)
    return acc


model = ResNet18(10)
ori_state_dict = torch.load("checkpoint_best.pt", map_location="cuda")
model.load_state_dict(ori_state_dict)
model = model.cuda().eval()
# print(model)

flops, params, results = count_flops_params(model, dummy_input)
print(f"FLOPs: {flops}, params: {params}")


# # cfg = [{'sparsity': 0.5, "op_names": ["layer4.1.conv1"], 'op_types': ['Conv2d']}]
# cfg = [{'sparsity': 0.5, "op_names": ["layer4.0.shortcut.0",
#                                       "layer4.0.conv2", "layer4.1.conv2"], 'op_types': ['Conv2d']}]
# # cfg = [{'sparsity': 0.5, 'op_types': ['Conv2d']}]
# pruner = FPGMPruner(model, cfg, dummy_input=torch.rand((4, 3, 32, 32)).cuda())
# pruner.compress()


# # 导出weight和mask
mask_path = 'mask.pth'
model_path = 'pruned.pth'
# pruner.export_model(model_path=model_path, mask_path=mask_path)

# # 这句话一定要在fix_mask_conflict之前，否则结果会错误
# pruner._unwrap_model()
# del pruner

# print("prune" + "*"*30)
# print(test(model))

# # 验证掩码冲突
fixed_mask_path = "fixed_mask.pth"
# new_mask = fix_mask_conflict(mask_path, model, dummy_input)
# torch.save(new_mask, fixed_mask_path)


model.load_state_dict(ori_state_dict)
# model = model.cuda().eval()
apply_compression_results(model, fixed_mask_path)
print("fixed" + "*"*30)
print(test(model))


model.load_state_dict(ori_state_dict)
# model = model.cuda().eval()
apply_compression_results(model, mask_path)
print("before_fixed" + "*"*30)
print(test(model))


ori_state_dict = torch.load("checkpoint_best.pt", map_location="cuda")
model.load_state_dict(ori_state_dict)
model = model.cuda().eval()

model_speedup = ModelSpeedup(model, dummy_input, mask_path, device)


model_speedup.speedup_model()
print("*"*30)
print(test(model))

flops, params, results = count_flops_params(model, dummy_input)
print(f"FLOPs: {flops}, params: {params}")
