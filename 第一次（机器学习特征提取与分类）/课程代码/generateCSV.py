#该代码我在课堂上没展示，功能是生成CSV文件夹
#如何使用：将不同的类别放在不同的文件夹中，文件夹的名字是标签的名字
import csv
import os
# 1. 创建文件对象
f = open('dataset.csv', 'w', newline='',encoding='utf-8')

# 2. 基于文件对象构建 csv写入对象
csv_writer = csv.writer(f)

# 3. 构建列表头
csv_writer.writerow(["path","classes","name"])
path = "F:\data_set\\flower_data\\flower_photos"#你的文件文件夹目录
dir = []
classes = []
for i in os.listdir(path):
    classes.append(i)
    # dir.append(os.path.join(path,i))
# 4. 写入csv文件内容

for i in range(len(classes)):
    for j in os.listdir(os.path.join(path,classes[i])):
        path1 = os.path.join(path,classes[i])
        csv_writer.writerow([os.path.join(path1,j), i,classes[i]])
# # 5. 关闭文件
f.close()