import os
from Predictor import Predictor
from PIL import Image
import shutil
from threading import Thread

with open("./log.txt", 'r') as f:
    # 记录每个类别数量的列表(总数, 普通人, 翻墙, 交递)
    allCount, normalCount, contactCount = tuple(map(int, f.read().split()))

# print(allCount, normalCount, climbCount, contactCount)
imageDir = './Images/'  # 待处理图片的路径
imageAfterDir = "./Images_after/"  # 处理完图片存放的位置
imageList = os.listdir(imageDir)  # 获取要处理的图片列表
# print(imageList)
proImageCount = len(os.listdir(imageAfterDir))  # 获取已处理的数量

predictor = Predictor()
proList = []


class MyThread(Thread):
    def __init__(self, fileList):
        super(MyThread, self).__init__()
        self.fileList = fileList

    def run(self):  # run函数为固定函数，表示当start是默认函数
        global proList
        global imageDir
        for file in self.fileList:
            if file == 'README.md':
                print("readme")
                continue
            imgDir = imageDir + file
            try:
                image = Image.open(imgDir)
            except FileNotFoundError:
                print('Open Error! Try again!')
                exit(0)
            else:
                resList = predictor.detectImage(image)
                # if not resList:
                #     continue
                resList.append(file)  # [[...], [...], [...], 'name']
                proList.append(resList)


if __name__ == '__main__':
    t = MyThread(imageList)
    t.start()
    while True:
        if len(proList) == 0:
            continue
        f0 = open("./Annotation/annotation_0.txt", 'a')
        f1 = open("./Annotation/annotation_1.txt", 'a')
        # f2 = open("./Annotation/annotation_2.txt", 'a')
        f = [f0, f1]
        ls = proList.pop(0)
        fname = ls[-1]
        ls = ls[:-1]
        count = [0, 0]

        for img in ls:  # img : [lmList, s_image[0]]
            img[1].show()
            anno = eval(input())
            if anno not in range(0, 2):
                continue
            f[anno].write(str(img[0].tolist()) + f" {anno}\n")
            count[anno] += 1
        allCount += sum(count)
        normalCount += count[0]
        # climbCount += count[1]
        contactCount += count[1]
        f0.close()
        f1.close()
        # f2.close()
        shutil.move(imageDir + fname, imageAfterDir + fname)
        # os.rename(imageAfterDir + fname, imageAfterDir + f"{proImageCount}.{fname.split('.')[-1]}")
        # proImageCount += 1
        with open("./log.txt", 'w') as f:
            f.write(f"{allCount} {normalCount} {contactCount}")
        print("save ,you can exit")
