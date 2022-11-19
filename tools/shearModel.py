# 用于分割出一张图像中的所有人体
import os

from PIL import Image

from YOLOX import YOLO

if __name__ == '__main__':
    yolo = YOLO()
    pTime = 0
    fundCoef = 0
    ids = 0
    imageDir = './img/PIC4/'  # 待处理图片的路径
    imageAfterDir = "./img/pic5/"  # 处理完图片存放的位置
    imageList = os.listdir(imageDir)  # 获取要处理的图片列表
    for i in imageList:
        # 一次检测一张图片
        img = imageDir + i
        try:
            image = Image.open(img)
        except FileNotFoundError:
            print('Open Error! Try again!')
            exit(0)
        else:
            r_image = yolo.detect_image(image)  # yolo检测，返回的是每个person图像
            print(type(r_image[0]))
            for s_image in r_image:  # 对每个person进行姿态识别，每个lmList里是各节点的坐标
                s_image[0].save(f"{imageAfterDir}{ids + fundCoef}.jpg")
                ids += 1
