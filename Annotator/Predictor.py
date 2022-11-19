import time

import cv2
import numpy as np
from PIL import Image

from MP import PoseDetector
from YOLOX import YOLO


class Predictor:
    mode = 0  # 0为单张图片, 1为视频

    def __init__(self):
        self.yolo = YOLO(path="../")
        self.detector = PoseDetector(True)

    def detectImage(self, img: Image) -> (Image, list[list[bool, list[int]]]):
        # 一次检测一张图片
        r_image = self.yolo.detect_image(img)  # yolo检测，返回的是[每个person图像,[四个边界]]
        resList = []
        for s_image in r_image:  # 对每个person进行姿态识别，每个lmList里是各节点的坐标
            cap = np.array(s_image[0], dtype=np.uint8)
            lmList = self.detector.drawPosition(cap, s_image[0])
            if lmList is None:
                continue
            temp = [lmList, s_image[0]]
            resList.append(temp)
        return resList


if __name__ == '__main__':
    predictor = Predictor()
    pTime = 0

    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except FileNotFoundError:
        print('Open Error! Try again!')
        exit(0)
    else:
        predictor.detectImage(image)
