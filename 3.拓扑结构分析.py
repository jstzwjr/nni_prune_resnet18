import torch
from cifar_resnet import ResNet18
from nni.compression.pytorch.utils.shape_dependency import ChannelDependency

'''
验证层之间的依赖关系
'''

model = ResNet18()
print(model)
dummy_input = torch.rand((1,3,32,32))
channel_depen = ChannelDependency(model, dummy_input)


channel_depen.export("outdir/dependency.csv")

