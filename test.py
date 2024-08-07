import torch
from net import MyAlexNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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
    ])
# 定义训练集和验证集，训练集和验证集的batch_size都是32，训练集随机翻转，验证集也随机翻转
train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# 定义一个GPU，如果GPU可用，则使用GPU，否则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义一个模型
model = MyAlexNet().to(device)

# 如果模型存在，则加载模型，否则训练模型
model.load_state_dict(torch.load("D:/AlexNet/save_model/best_model.pth")) # todo

# 获取预测结果
# 定义一个类别列表
classes = [
    "cat",
    "dog",
]

# 定义一个ToPILImage对象，并把张量转化为图片格式，并显示
show = ToPILImage()

# 进入到验证阶段，验证模型，并输出预测结果，以及实际结果，如果预测正确，则输出"Correct"，否则输出"Wrong"，并显示图片
model.eval()
for i in range(2490,2510):
    x, y = val_dataset[i][0], val_dataset[i][1]
    show(x).show()
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True).to(device)
    x = torch.tensor(x).to(device)
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}", Actual:"{actual}"')
