import torch
import os
from PIL import Image
from torchvision import datasets, models, transforms

#----------------------------------------------------------------------------------
#       汽车数据集，7分类
#       数据集地址：<https://www.kaggle.com/datasets/kshitij192/cars-image-dataset>
#----------------------------------------------------------------------------------

#数据集地址
train_dataset = "CarsDataset\\train"
test_dataset = "CarsDataset\\test"

#获取图片路径
file_train = []
for d in os.listdir(train_dataset):
    d = os.path.join(train_dataset, d)
    for _ in os.listdir(d):
        _ = os.path.join(d, _)
        file_train.append(_)

file_test = []
for d in os.listdir(test_dataset):
    d = os.path.join(test_dataset, d)
    for _ in os.listdir(d):
        _ = os.path.join(d, _)
        file_test.append(_)

# train_list, val_list = train_test_split(file_train, test_size=0.2)

# 数据分割，把图片转化成112*112大小
train_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomResizedCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomResizedCrop(112),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


class dataset(torch.utils.data.Dataset):

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    # 数据长度
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    # 载入一张图片返回的东西，返回一个img和一个label
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        label = img_path.split('\\')[2]
        if label == 'Audi':
            label = 0
        if label == 'Hyundai Creta':
            label = 1
        if label == 'Mahindra Scorpio':
            label = 2
        if label == 'Rolls Royce':
            label = 3
        if label == 'Swift':
            label = 4
        if label == 'Tata Safari':
            label = 5
        if label == 'Toyota Innova':
            label = 6

        return img_transformed, label

#自定义dataloader方法
def MyDataloader(batch_size):
    train_data = dataset(file_train, transform=train_transforms)
    test_data = dataset(file_test, transform=test_transforms)
    #shuffle：打乱
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    return train_loader,test_loader

