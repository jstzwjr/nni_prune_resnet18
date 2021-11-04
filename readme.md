#### 一、项目描述

该项目是nni的简单使用，主要涉及模型训练（resnet18、cifar10）、灵敏度分析、拓扑结构分析（通道依赖）、基本剪枝、模型加速等。

#### 二、脚本介绍

##### 1.train_cifar10.py

该脚本主要实现功能为使用resnet18训练cifar10数据集。

需要注意torchvision和cifar10_resnet的resnet18的几点区别：

​	①.torchvision的resnet18的conv1卷积核为7x7，而cifar10_resnet18的conv1为3x3；

​	②.torchvision的layer2的stride为2，而cifar10_resnet18的layer2的stride为1；

​	③.torchvision的layer1之前有maxpool操作。

如果使用torchvision的resnet18进行相同的操作，训练出来的精度要低5个点左右，最终模型精度为91.34%。

##### 2.sensitivity.py

该脚本主要实现功能为模型灵敏度分析，对每个层分别进行一定比例范围内的剪枝，然后对模型精度进行测试，生成结果文件保存到outdir。

```python
# 注意，由于test函数的返回值范围是0~100，所以early_stop_value范围也是0~100
# SensitivityAnalysis本质上是对所有的层（目前只支持conv2d和conv1d）进行prune操作，再分别根据val_func函数计算分析指标
s_analyzer = SensitivityAnalysis(model=model, val_func=test, early_stop_mode="minimize", early_stop_value=50, prune_type="fpgm")

# 可以通过specified_layer参数指定只分析哪些层的灵敏度，如["conv1", "layer1.0.conv1"]
sensitivity = s_analyzer.analysis(val_args=[model],specified_layers=["layer4.1.conv1"])
```

```python
# 特别注意！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# SensitivityAnalysis分析的结果其实并不是准确的，因为经过剪枝后的模型验证时，本质上是weight乘上mask，类似bn层的对应通道还是参与的计算，导致结果有一定偏差
# 真实的结果应该是经过modelspeedup之后的结果
os.makedirs("outdir", exist_ok=True)
s_analyzer.export(os.path.join("outdir", "fpgm.csv"))
```

##### 3.拓扑结构分析.py

该脚本主要作用是分析层之间通道依赖关系，如果层之间有通道依赖关系，则不能按照指定的稀疏度进行剪枝。

##### 4.1.prune.py

该脚本主要功能有：

暴力剪枝（对所有的卷积层按照指定的稀疏度剪枝）

```python
cfg = [{'sparsity': 0.3, 'op_types': ['Conv2d']}]
pruner = FPGMPruner(model, cfg, dummy_input=torch.rand((4,3,32,32)).cuda())
pruner.compress()
```

只剪某一层

```python
cfg = [{'sparsity': 0.5, "op_names": ["layer4.1.conv2"], 'op_types': ['Conv2d']}]
pruner = FPGMPruner(model, cfg, dummy_input=torch.rand((4,3,32,32)).cuda())
pruner.compress()
```

剪某几层，例子中的几个层是有通道依赖关系的，所以最终剪枝比例可能没有0.5

```python
cfg = [{'sparsity': 0.5, "op_names": ["layer4.0.shortcut.0","layer4.0.conv2","layer4.1.conv2"], 'op_types': ['Conv2d']}]
pruner = FPGMPruner(model, cfg, dummy_input=torch.rand((4,3,32,32)).cuda())
pruner.compress()
```

最后导出model、mask和onnx到指定路径。

##### 4.2.掩码冲突.py

掩码冲突校验，如果对有通道依赖关系的层进行剪枝，需只修剪公共的部分

```python
new_mask = fix_mask_conflict(mask_path, model, dummy_input)
torch.save(new_mask, fixed_mask_path)
```

##### 4.3.模型加速.py

进行模型加速，生成剪枝后的模型，并导出到onnx。如果不保存onnx，每次加载模型都需要进行一次speedup操作。也可保存成torchscript。

```shell
4次测试结果分别是

91.34  原始模型结果

66.39  经过剪枝（0.5，应该是256个通道）的结果

90.28  经过通道冲突fix之后的结果（只剪了110个通道）

90.78  经过加速的模型，除了weight乘了mask，bn层对应的通道也剪了，因此和结果3会有一定偏差
```

##### 4.4.验证剪枝后的模型.py

计算剪枝前后参数量、计算量等。