"""
trainRotNet -

Author:霍畅
Date:2024/6/15
"""
import os
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import DataLoader

# 数据集路径
# dataroot = "..\data\VOCtrainval_11-May-2012"
dataroot = "..\data"
# 文件保存路径
current = datetime.now()
run_time = current.strftime("%Y_%m_%d_%H_%M")
tb_dir = os.path.join('./logs/Rot', run_time)
writer = SummaryWriter(log_dir=tb_dir)
model_dir = os.path.join('./models', f"RN_{run_time}.pth")
# 是否保存模型
save = 1
# 是否继续训练
reload = 0
if reload:
    model_load_dir = './models/RotNet_pretrained02.pth'
    # 根据具体情况修改
    last_epoch = 16
    lase_iter = last_epoch * 1285
else:
    model_load_dir = ''
    last_epoch = 0
    lase_iter = 0


def test(net, testloader, criterion):
    total_loss = 0.0
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_and_test(net, trainloader, testloader, criterion, optimizer, epochs=10):
    best_accuracy = 0.0
    if save:
        torch.save(net.state_dict(), model_dir)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        total = 0
        batches = len(trainloader)
        correct = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            if i % 100 == 99:
                train_accuracy = 100 * correct / total
                print(
                    f'[{epoch + 1}/{epochs}, {i + 1}/{batches}] loss: {running_loss / total:.3f} accuracy:{train_accuracy:.2f}%')
                iter = lase_iter + epoch * batches + (i + 1)
                writer.add_scalar('Loss/train', running_loss / total, iter)
                writer.add_scalar('Accuracy/train', train_accuracy, iter)
                running_loss = 0.0
                correct = 0
                total = 0
        test_loss, test_accuracy = test(net, testloader, criterion)
        best_accuracy = max(best_accuracy, test_accuracy)
        print(f'Test Loss: {test_loss:.3f}   Accuracy of epoch {epoch + 1}: {test_accuracy:.2f}%,')
        writer.add_scalar('Loss/test', test_loss, last_epoch + epoch + 1)
        writer.add_scalar('Accuracy/test', test_accuracy, last_epoch + epoch + 1)
        if best_accuracy == test_accuracy and save:
            torch.save(net.state_dict(), model_dir)
    writer.close()


if __name__ == '__main__':
    # 设置超参数
    learning_rate = 0.001
    batch_size = 32
    total_epoch = 100
    weight_decay = 0.01
    num_classes = 4
    # 选择网络架构 resnet18
    net = models.resnet18()
    # 修改最后一层网络架构，并随机初始化
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    if reload:
        net.load_state_dict(torch.load(model_load_dir))
    else:
        pass

    # 确保使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net = net.to(device)
    print("Network = ", net)

    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集和测试集
    trainset, testset = DataLoader.LoadRotationDataset2(dataroot, batch_size, transform)
    classes = ('0', '90', '180', '270')
    print("Data loaded successfully")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练网络
    print("Start Training")
    t1 = time.time()
    train_and_test(net, trainset, testset, criterion, optimizer, epochs=total_epoch)
    t2 = time.time()
    print('Finished Training')
    print(f"Total Time =   {t2 - t1:.3f} s")
    print(f"Average Time = {(t2 - t1) / total_epoch:.3f} s/epoch")
