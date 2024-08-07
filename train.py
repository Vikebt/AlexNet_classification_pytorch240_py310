import os
import torch
import numpy as np
from torch import nn
from net import MyAlexNet
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据集路径
ROOT_TRAIN = r'D:/AlexNet/data/train' # todo
ROOT_TEST = r'D:/AlexNet/data/val' # todo



# 将图像的像素值归一化到【-1， 1】之间，并且将图像的通道数调整为3，因为AlexNet是3通道的
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# 定义训练集和验证集的转换，
# 训练集和验证集的图像尺寸都调整为224*224，因为AlexNet的输入是224*224大小的
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])
# 定义训练集和验证集，训练集和验证集的batch_size都是32，训练集随机翻转，验证集也随机翻转
train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# 定义一个GPU，如果GPU可用，则使用GPU，否则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义一个模型
model = MyAlexNet().to(device)

# 定义一个损失函数
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器，学习率为0.01，动量为0.9，
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 定义一个学习率调度器，学习率每隔10轮变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    # 将模型转化为训练模型
    model.train()
    # 初始化损失和准确率
    loss, current, n = 0.0, 0.0, 0
    # 开始训练,每次训练一个batch的数据
    for batch, (x, y) in enumerate(dataloader):
        # 前向传播，将数据和标签转换成torch.tensor再转移到GPU
        image, y = x.to(device), y.to(device)
        # 神经网络模型训练，计算输出
        output = model(image)
        # 计算损失
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        # 计算准确率
        cur_acc = torch.sum(y==pred) / output.shape[0]

        # 反向传播，优化器更新参数为0
        optimizer.zero_grad()
        # 反向传播，计算损失函数
        cur_loss.backward()
        # 更新梯度
        optimizer.step()
        # 更新一个批次内的总损失和总准确率
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n+1
    # 更新一轮内的平均损失和平均准确率
    train_loss = loss / n
    train_acc = current / n
    print('train_loss' + str(train_loss))
    print('train_acc' + str(train_acc))
    return train_loss, train_acc

# 定义验证函数,验证函数和训练函数基本类似，只是将优化器换成验证优化器，
# 验证函数中没有反向传播、学习率调度器、优化器更新参数，因为验证函数中不需要更新参数
# 验证函数中没有更新损失和准确率，因为验证函数中不需要更新损失和准确率
# 验证函数中没有训练轮数，因为验证函数中不需要训练轮数
def val(dataloader, model, loss_fn):
    # 将模型转化为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n
    print('val_loss' + str(val_loss))
    print('val_acc' + str(val_acc))
    return val_loss, val_acc

# 定义画图函数，画出训练集和验证集的loss值
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("训练集和验证集loss值对比图")
    plt.show()
# 定义画图函数，画出训练集和验证集的acc值
def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("训练集和验证集acc值对比图")
    plt.show()


# 开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

# 定义一个训练轮数，这里设置的是10轮，如果想要继续训练，则将10改为更大的数
epoch = 10
min_acc = 0
# 开始训练，每轮训练都会更新学习率
#
for t in range(epoch):
    # 每十步更新学习率
    lr_scheduler.step()
    print(f"epoch{t+1}\n-----------")
    # 开始训练，训练一个轮次，返回训练损失和准确率
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    # 开始验证，验证一个轮次，返回验证损失和准确率
    val_loss, val_acc = val(val_dataloader, model, loss_fn)

    # 记录训练损失和准确率
    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    # 保存最好的模型权重,如果验证准确率大于最小值，则保存模型
    if val_acc >min_acc:
        folder = 'save_model'
        # 如果文件夹不存在，则创建文件夹
        if not os.path.exists(folder):
            os.mkdir('save_model')
        # 更新准确率，保存模型
        min_acc = val_acc
        print(f"save best model, 第{t+1}轮")
        # 保存最好的模型权重文件
        torch.save(model.state_dict(), 'save_model/best_model.pth') # todo
    # 保存最后一轮的权重文件
    if t == epoch-1:
        torch.save(model.state_dict(), 'save_model/last_model.pth') # todo
# 画出训练集和验证集的loss值和acc值
matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)
print('Done!')
