#获取图像的RGB直方图
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people

#获取数据集，min_faces_per_person=70表示获取图片大于70张的人物，
# resize=n表示获取原图大小的几倍0<n<1,color=True表示获取彩色图像，
# False表示灰度图像
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=2,color=True)
#获取图像特征
X = lfw_people.data
#获取原图像特征个数
n_features = X.shape[1]
#预测的标签
y = lfw_people.target
#数据集中的人名
target_names = lfw_people.target_names
#一共有多少类人
n_classes = target_names.shape[0]
print(lfw_people.images.shape)
#获取图片image[3]表示获取第三张图，可以获取第二或第一也可以
img = lfw_people.images[3]
#显示原图
plt.imshow(np.uint8(img))
plt.show()
#显示该图片的RGB直方图
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()