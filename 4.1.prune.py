import torch
from torchvision.models import resnet18
import torch.nn as nn
from nni.algorithms.compression.pytorch.pruning import FPGMPruner
from cifar_resnet import ResNet18
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import logging

logger = logging.getLogger(__name__)

device = "cuda:0"
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

    # print('Test Loss: {:.6f}  Accuracy: {}%\n'.format(
    #     test_loss, acc))
    return acc


model = ResNet18(10)
ori_state_dict = torch.load("checkpoint_best.pt", map_location="cuda")
model.load_state_dict(ori_state_dict)
model = model.cuda().eval()

# 暴力剪枝  对所有的卷积层按照指定的稀疏度剪枝
# cfg = [{'sparsity': 0.3, 'op_types': ['Conv2d']}]
# pruner = FPGMPruner(model, cfg, dummy_input=torch.rand((4,3,32,32)).cuda())
# pruner.compress()


# 只剪某一层
# cfg = [{'sparsity': 0.5, "op_names": ["layer4.1.conv2"], 'op_types': ['Conv2d']}]
# pruner = FPGMPruner(model, cfg, dummy_input=torch.rand((4,3,32,32)).cuda())
# pruner.compress()

# 剪某几层     例子中的几个层是有通道依赖关系的，所以最终剪枝比例可能没有0.5
cfg = [{'sparsity': 0.5, "op_names": ["layer4.0.shortcut.0",
                                      "layer4.0.conv2", "layer4.1.conv2"], 'op_types': ['Conv2d']}]
pruner = FPGMPruner(model, cfg, dummy_input=torch.rand((4, 3, 32, 32)).cuda())
pruner.compress()


print(test(model))
# 将剪枝后的权重和mask保存
pruner.export_model("model.pth", "mask.pth", "res.onnx", input_shape=(4, 3, 32, 32), device=device)


# # 根据灵敏度分析结果进行剪枝
# # cfg = [{'sparsity': 0.3, 'op_names': ["conv1"], 'op_types': ['Conv2d']}]
# for val1 in np.arange(0.1, 1.0, 0.1):
#     for val2 in np.arange(0.1, 1.0, 0.1):
#         cfg = [{"sparsity": val2, "op_names": ["layer4.1.conv2"], 'op_types': ['Conv2d']},
#                {"sparsity": val1, "op_names": ["layer4.1.conv1"], 'op_types': ['Conv2d']}]
#         # cfg = [{"sparsity": val2, "op_names": ["layer4.1.conv2"], 'op_types': ['Conv2d']},]
#         pruner = FPGMPruner(model, cfg, dummy_input=torch.rand((4,3,32,32)).cuda())
#         pruner.compress()
#         print(val1, val2, test(model))
#         pruner._unwrap_model()
#         del pruner
#     # 这句话一定要加，否则会在之前剪枝的基础上进行剪枝
#     model.load_state_dict(ori_state_dict)
