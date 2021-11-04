import torch
import numpy as np
from nni.compression.pytorch.utils.counter import count_flops_params
from cifar_resnet import ResNet18
import torch.nn as nn
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_workers = 16
torch.set_num_threads(num_workers)


def test(model, valid_dataloader):
    model.eval()

    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(valid_dataloader)):
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)

    valid_loss = np.array(loss_list).mean()
    valid_acc = np.array(acc_list).mean()

    return valid_loss, valid_acc


def train():
    # torchvision和cifa10_resnet的区别：
    # 1.torchvision的resnet18的conv1卷积核为7x7，而cifar10_resnet18的conv1为3x3；
    # 2.torchvision的layer2的stride为2，而cifar10_resnet18的layer2的stride为1；
    # 3.torchvision的layer1之前有maxpool操作。
    model = ResNet18(num_classes=10)
    model = model.to(device)

    print(model)

    # check model FLOPs and parameter counts with NNI utils
    dummy_input = torch.rand([1, 3, 32, 32]).to(device)

    flops, params, results = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")

    train_dataloader = torch.utils.data.DataLoader(
        CIFAR10('data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]), download=True), batch_size=512, num_workers=num_workers)
    valid_dataloader = torch.utils.data.DataLoader(
        CIFAR10('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]), download=True), batch_size=512, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # lr_policy = torch.optim.lr_scheduler.StepLR(optimizer, 70, 0.1)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_valid_acc = 0

    for epoch in range(100):
        print('Start training epoch {}'.format(epoch))
        loss_list = []
        # train
        model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        lr_policy.step()
        print(lr_policy.get_lr()[0])

        # validation
        valid_loss, valid_acc = test(model, valid_dataloader)
        train_loss = np.array(loss_list).mean()
        print('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}'.format
              (epoch, train_loss, valid_loss, valid_acc))

        # save
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'checkpoint_best.pt')


if __name__ == '__main__':
    train()

    # 测试代码
    # model = ResNet18(num_classes=10).to(device)
    # model.load_state_dict(torch.load("checkpoint_best.pt", map_location=device))
    # model.eval()

    # valid_dataloader = torch.utils.data.DataLoader(
    #     CIFAR10('data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #     ]), download=True), batch_size=512, num_workers=num_workers)
    # valid_loss, valid_acc = test(model, valid_dataloader)
    # print(valid_acc, valid_loss)
