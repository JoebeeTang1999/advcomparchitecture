import random
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
import torch
import torchvision
from torchvision import transforms, datasets

data_transform = {
    # 训练中的数据增强和归一化
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 随机裁剪
        transforms.RandomHorizontalFlip(), # 左右翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 均值方差归一化
    ])
}

# from main import data_transform

# load the saved model./
# with open('best_model.pkl', 'rb') as f:
path='best_model.pkl'
model = torch.load(path)# 图片路径
# img_path = input()
# r'./data/images/144.jpg'

# 打开图像

randnum = random.randint(100,230)
randnum = str(randnum)
img_path = "D:\\PythonProjects\\ResNet Cake Classification\\data\\images\\157.jpg"
img_path = img_path.replace("157",randnum)
img = Image.open(img_path)
# img = Image.open(r'./data/images/157.jpg')
# 绘制图像
plt.imshow(transforms.ToTensor()(img).permute(1,2,0))
plt.show()

# # 对图像进行变换
img = data_transform['train'](img)

# 将图像升维，增加batch_size维度
img = torch.unsqueeze(img, dim=0)
# predict using the loaded model

# 加载模型
model = torchvision.models.resnet50(pretrained=False) # 加载预训练好的ResNet50模型
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 3),nn.Sigmoid())
model.load_state_dict(torch.load(path))

# 获取预测结果
model.eval()
pred = model(img) > 0.5

dic = {False: 'no', True: 'yes'}

# 预测输出
print('【预测cream结果分类】：%s' % dic[pred[0][0].item()])
print('【预测fruits结果分类】：%s' % dic[pred[0][1].item()])
print('【预测sprinkle_toppings结果分类】：%s' % dic[pred[0][2].item()])