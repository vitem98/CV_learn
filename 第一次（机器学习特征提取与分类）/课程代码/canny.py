import cv2
from matplotlib import pyplot as plt

######################################################################
#该模块是展示不同的高低阈值对边缘检测的影响
#####################################################################
#读取图片
img = cv2.imread('canny1.png',1)
#分别对图片娶不同高低阈值
edges = cv2.Canny(img,100,200)
edges1 = cv2.Canny(img,50,150)
#输出图片
plt.subplot(131),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image(100,200)'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(edges1,cmap = 'gray')
plt.title('Edge Image(50,150)'), plt.xticks([]), plt.yticks([])
plt.show()

######################################################################
#该模块噪声对canny的影响以及利用高斯模糊对噪声图片的减噪来提高canny的使用
######################################################################
# 读入图像
lenna = cv2.imread("canny2.png", 0)
# 图像降噪（不同的高斯模糊）
# lenna = cv2.GaussianBlur(lenna, (5, 5), 0)
lenna = cv2.GaussianBlur(lenna, (9, 9), 0)
# Canny边缘检测
edges = cv2.Canny(lenna, 100, 150)
#输出图片
plt.subplot(121),plt.imshow(lenna,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

