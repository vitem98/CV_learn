import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from Attention import se_block, SpatialAttention
from Dataloader import MyDataloader


device = "cuda" if torch.cuda.is_available() else "cpu"
#----------------------------------------------------------------
#   超参数设置
#   phi：为是否使用注意力机制，默认为0不使用
#       1：使用通道注意力（）
#       2：使用空间注意力
#   dataset：选择数据集
#       0：FashionMNIST数据集（默认）
#       1: 汽车数据集
#       《注意》：当选用汽车数据集时，模型的Fc1那里输入输出tensor需要自己改
#----------------------------------------------------------------
learning_rate = 1e-3
batch_size = 64
epochs = 10
batchsize = 64
phi = 2
dataset = 1

#--------------------------------------------------------------------------------------------------------
#    数据导入模块，用了torchvision的FashionMNIST数据集
#    数据集地址：<https://github.com/zalandoresearch/fashion-mnist>
#    数据集介绍：图片大小为28*28，通道数为1
#    参数介绍：root：数据路径
#            train：是否训练集
#            download：没有数据集是否下载
#            transform：数据格式
#    详情请看链接：<https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataloader>
#--------------------------------------------------------------------------------------------------------
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#数据装入dataloader
if dataset == 0:
    train_dataloader = DataLoader(training_data, batch_size=batchsize)
    test_dataloader = DataLoader(test_data, batch_size=batchsize)
elif dataset == 1:
    train_dataloader,test_dataloader = MyDataloader(batch_size=batchsize)
#-------------------------------------------------------------------
#     网络模型基本块（block）
#     先卷积、初始化、激活函数，先用1*1卷积降低通道数，再用3*3卷积增大感受野
#     整个模型用到了残差模块
#     该模块输入的shape和输出的shape一致
#-------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        # 1*28*28-->8*28*28
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        #8*28*28-->16*28*28
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out

#----------------------------------------------------------------------
#       自定义网络模块
#   ·   原始结构模型：BasicBlock-->BasicBlock-->Flatten-->FC-->relu-->FC
#       可以自由组合搭配
#----------------------------------------------------------------------
class JeffNetwork(nn.Module):
    def __init__(self,phi = 0, dataset = 0):
        super(JeffNetwork, self).__init__()
        self.phi = phi
        self.dataset = dataset
        if dataset == 0:
            self.block1 = BasicBlock(1, [8, 16])
            # 全连接层输出模块,
            self.Fc1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28 * 28 * 16, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )
        if dataset == 1:
            self.conv1 = nn.Conv2d(3,16,kernel_size=1, stride=1, padding=0)
            self.block1 = BasicBlock(16, [8, 16])
            # 当选择汽车数据集时，在卷积层后面加个pooling操作，减少训练场数量
            self.Fc2 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(56 * 56 * 16, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        self.block2 = BasicBlock(16, [8,16])

        #注意力机制模块
        if phi == 1:
            self.senet1 = se_block(16)
            self.senet2 = se_block(16)
        if phi == 2:
            self.sam1 = SpatialAttention(kernel_size=3)
            self.sam2 = SpatialAttention(kernel_size=3)


    def forward(self, x):
        # 1*28*28-->16*28*28
        if self.dataset == 1:
            x = self.conv1(x)
        # 1*28*28-->16*28*28
        x = self.block1(x)
        #------------------
        #   注意力模块
        #------------------
        # 16*28*28-->16*28*28
        if self.phi ==1:
            x = self.senet1(x)
        if self.phi == 2:
            x = self.sam1(x)
        # 16*28*28-->16*28*28
        x = self.block2(x)
        #------------------
        #   注意力模块
        #------------------
        # 16*28*28-->16*28*28
        if self.phi == 1:
            x = self.senet1(x)
        if self.phi == 2:
            x = self.sam1(x)

        # 16*28*28-->10
        if self.dataset == 0:
            x = self.Fc1(x)
        if self.dataset == 1:
            x = self.Fc2(x)

        return x

#----------------------------------
#       模型初始化
#       loss初始化
#       优化器初始化，一般用adam和sgd
#----------------------------------
model = JeffNetwork(phi=phi,dataset=dataset).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#----------------------------------------------------------------------------
#       训练函数与测试函数定义，该实验没有用到验证集，直接就是训练集训练测试集测试精度
#       测试的准确率：测试正确的结果/测试个数
#----------------------------------------------------------------------------
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # 计算预测值和loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #得到的是一个10*batchsize大小向量，对应的10个分类，取数值最大的为该预测类
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#-------------------------------------------------------
#       开始迭代训练
#       该模型占用显存约为1.1G左右，一般在自己笔记本就可以跑
#-------------------------------------------------------
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
