from nni.compression.pytorch.speedup import ModelSpeedup
from cifar_resnet import ResNet18
import torch
from nni.compression.pytorch import apply_compression_results
from time import time
from nni.compression.pytorch.utils.counter import count_flops_params
from torchvision import datasets
from torchvision.transforms import transforms

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
    return acc


mask_path = 'mask.pth'
fixed_mask_path = "fixed_mask.pth"


model = ResNet18(10)
ori_state_dict = torch.load("checkpoint_best.pt", map_location="cuda")
model = model.cuda().eval()
model.load_state_dict(ori_state_dict)
# 原始模型测试
print(test(model))


# 剪枝模型测试（只是某些weight置零，bn没有剪通道，导致某些为0的输出变为非0，结果有偏差）
# apply_compression_results函数的作用本质上就是给权重乘上了mask
apply_compression_results(model, mask_path)
print(test(model))


# 去除冲突掩码后测试结果，存在和上面一样的问题
# 需要重新加载初始权重
model.load_state_dict(ori_state_dict)
apply_compression_results(model, fixed_mask_path)
print(test(model))


# 模型加速
# 需要重新加载权重
model.load_state_dict(ori_state_dict)
dummy_input = torch.rand((1000, 3, 32, 32)).cuda()

model_speedup = ModelSpeedup(model, dummy_input, mask_path)
model_speedup.speedup_model()
print(test(model))

torch.onnx.export(
        model,
        torch.rand((4, 3, 32, 32)).cuda(),
        "speedup.onnx",
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=False,
        opset_version=11)

'''
4次测试结果分别是
91.34   原始模型结果
66.39   经过剪枝（0.5，应该是256个通道）的结果
90.28   经过通道冲突fix之后的结果（只剪了110个通道）
90.78   经过加速的模型，除了weight乘了mask，bn层对应的通道也剪了，因此和结果3会有一定偏差
'''
