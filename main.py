import torchvision
from torch import nn
import numpy as np
import pandas as pd
import os
import json
import pickle

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt

# 加载数据集，自己重写DataSet类
class dataset(Dataset):
    # image_dir为数据目录，label_file，为标签文件
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir # 图像文件所在路径
        self.label_file = pd.read_csv(label_file) # 图像对应的标签文件
        self.transform = transform # 数据转换操作

    # 加载每一项数据
    def __getitem__(self, idx):
        # 每个图片，其中idx为数据索引
        img_name = os.path.join(self.image_dir, '%.3d.jpg' % (idx + 1)) # 加载每一张照片
        image = Image.open(img_name)

        # 对应标签
        labels = (self.label_file[['cream', 'fruits', 'sprinkle_toppings']] == 'yes').astype(int).values[idx, :]

        if self.transform:
            image = self.transform(image)

        # 返回一张照片，一个标签
        return image, labels

    # 数据集大小
    def __len__(self):
        return (len(self.label_file))

image_dir = './data/images' # 数据集路径
label_file = './data/cake_annotated.csv' # 标签位置
epochs = 10
lr = 0.003
batch_size = 32
save_path = './best_model.pkl'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 1.数据转换
data_transform = {
    # 训练中的数据增强和归一化
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 随机裁剪
        transforms.RandomHorizontalFlip(), # 左右翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 均值方差归一化
    ])
}

# 2.形成训练集
train_dataset = dataset(image_dir, label_file, data_transform['train'])

# 3.形成迭代器
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size,
                                           True)

print('using {} images for training.'.format(len(train_dataset)))

# 自定义损失函数，需要在forward中定义过程
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    # 参数为传入的预测值和真实值，返回所有样本的损失值，自己只需定义计算过程，反向传播PyTroch会自动记录，最好用PyTorch进行计算
    def forward(self, pred, label):
        # pred：[32, 3] label：[32, 3] 第一维度是样本数
        # 由于是二分类，使用BCE损失
        return F.binary_cross_entropy(pred, label)

# 3.加载ResNet50模型
model = torchvision.models.resnet50(pretrained=True) # 加载预训练好的ResNet50模型

# 4.冻结模型参数
for param in model.parameters():
    param.requires_grad = False

# 5.修改最后一层的全连接层
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 3),
    nn.Sigmoid())

# 6.将模型加载到cpu中
model = model.to('cpu')

criterion = MyLoss() # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 优化器

# 7.模型训练
best_acc = 0 # 最优精确率
best_model = None # 最优模型参数

for epoch in range(epochs):
    model.train()
    running_loss = 0 # 损失

    epoch_acc_cream = 0  # 每个epoch的准确率
    epoch_acc_fruits = 0  # 每个epoch的准确率
    epoch_acc_sprinkle_toppings = 0  # 每个epoch的准确率

    epoch_acc_count_cream = 0  # 每个epoch训练的样本数
    epoch_acc_count_fruits = 0  # 每个epoch训练的样本数
    epoch_acc_count_sprinkle_toppings = 0  # 每个epoch训练的样本数

    train_count = 0  # 用于计算总的样本数，方便求准确率

    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data
        optimizer.zero_grad()
        output = model(images.to(device))
        loss = criterion(output, labels.float().to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

        # 计算每个epoch正确的个数
        epoch_acc_count_cream += ((output > 0.5).int() == labels.float().to(device)).sum(axis=0)[0]
        epoch_acc_count_fruits += ((output > 0.5).int() == labels.float().to(device)).sum(axis=0)[1]
        epoch_acc_count_sprinkle_toppings += ((output > 0.5).int() == labels.float().to(device)).sum(axis=0)[2]

        train_count += len(images)

    # 每个epoch对应的准确率
    epoch_acc_cream = epoch_acc_count_cream / train_count
    epoch_acc_fruits = epoch_acc_count_fruits / train_count
    epoch_acc_sprinkle_toppings = epoch_acc_count_sprinkle_toppings / train_count

    # 打印信息
    print("【EPOCH: 】%s" % str(epoch + 1))
    print("训练损失为%s" % str(running_loss))
    print("cream训练精度为%s" % (str(epoch_acc_cream.item() * 100)[:5]) + '%')
    print("fruits训练精度为%s" % (str(epoch_acc_fruits.item() * 100)[:5]) + '%')
    print("sprinkle_toppings训练精度为%s" % (str(epoch_acc_sprinkle_toppings.item() * 100)[:5]) + '%')

    if epoch_acc_cream > best_acc:
        best_acc = epoch_acc_cream
        best_model = model.state_dict()

    # 在训练结束保存最优的模型参数
    if epoch == epochs - 1:
        # 保存模型
        torch.save(best_model, save_path,_use_new_zipfile_serialization=False)

print('Finished Training')

print("请输入要检测的蛋糕图片")

# 数据变换
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 图片路径
img_path = input()
# r'./data/images/144.jpg'

# 打开图像
img = Image.open(img_path)

# 绘制图像
plt.imshow(transforms.ToTensor()(img).permute(1,2,0))
plt.show()

# 对图像进行变换
img = data_transform(img)

# 将图像升维，增加batch_size维度
img = torch.unsqueeze(img, dim=0)

# 获取预测结果
model.eval()
pred = model(img) > 0.5

dic = {False: 'no', True: 'yes'}

# 预测输出
print('【预测cream结果分类】：%s' % dic[pred[0][0].item()])
print('【预测fruits结果分类】：%s' % dic[pred[0][1].item()])
print('【预测sprinkle_toppings结果分类】：%s' % dic[pred[0][2].item()])
