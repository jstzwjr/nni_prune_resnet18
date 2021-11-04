from nni.compression.pytorch.utils.sensitivity_analysis import SensitivityAnalysis
import torch
from torchvision import datasets, transforms
from cifar_resnet import ResNet18
import os

device = "cuda:0"


def test(model):
    criterion = torch.nn.CrossEntropyLoss()
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True), batch_size=512, num_workers=16)
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


model = ResNet18(num_classes=10).to(device)
model.load_state_dict(torch.load("checkpoint_best.pt", map_location=device))
model.eval()

# print(test(net))

# 注意，由于test函数的返回值范围是0~100，所以early_stop_value范围也是0~100
# SensitivityAnalysis本质上是对所有的层（目前只支持conv2d和conv1d）进行prune操作，再分别根据val_func函数计算分析指标
s_analyzer = SensitivityAnalysis(model=model, val_func=test, early_stop_mode="minimize", early_stop_value=50,
                                 prune_type="fpgm")


# 可以通过specified_layer参数指定只分析哪些层的灵敏度，如["conv1", "layer1.0.conv1"]
sensitivity = s_analyzer.analysis(val_args=[model],specified_layers=["layer4.1.conv1"])


# 特别注意！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# SensitivityAnalysis分析的结果其实并不是准确的，因为经过剪枝后的模型验证时，本质上是weight乘上mask，类似bn层的对应通道还是参与的计算，导致结果有一定偏差
# 真实的结果应该是经过modelspeedup之后的结果
os.makedirs("outdir", exist_ok=True)
s_analyzer.export(os.path.join("outdir", "fpgm.csv"))
