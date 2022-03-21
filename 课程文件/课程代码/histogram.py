#生成图片的灰度直方图
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people

#获取数据集，min_faces_per_person=70表示获取图片大于70张的人物，
# resize=n表示获取原图大小的几倍0<n<1,color=True表示获取彩色图像，
# False表示灰度图像
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
#获取图像特征
X = lfw_people.data
#获取图片，img的shape为一维的X[n],n表示第几章图片
img = X[0]
##显示该图片的RGB直方图
color = ('b','g','r')
#bins为横坐标，灰度图分为几份
hist, bins = np.histogram(img, bins=50)
plt.plot(hist)
plt.xlim([0,50])
plt.show()