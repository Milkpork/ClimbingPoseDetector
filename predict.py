import math

import cv2
import numpy as np
from PIL import Image
import torch.nn as nn

from DenseNet import DenseNet
from MP import PoseDetector
from YOLOX import YOLO
from utils import PrimeDic, Processor


def distance(node1, node2):
    try:
        res = math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
        return res
    except TypeError:
        raise ValueError(f"node1:{node1},node2:{node2} value error")


class Predictor:
    confidence = 0.5
    distance = 0
    nearDistance = 0
    climbTopLine = 0  # 超过这条线则在爬墙
    wallLine = 0
    cuda = False

    def __init__(self):
        self.yolo = YOLO(use_cuda=self.cuda)
        self.detector = PoseDetector(mode=True)
        self.desnet = DenseNet(path='./', cuda=self.cuda)
        self.pTime = 0
        self.generate(self.cuda)

    def generate(self, cuda=False):
        self.desnet = self.desnet.eval()
        if cuda:
            self.desnet = nn.DataParallel(self.desnet)
            self.desnet = self.desnet.cuda()

    def detectImage(self, img: Image) -> (Image, list[list[bool, list[int]]]):
        # 一次检测一张图片
        r_image, wall_image = self.yolo.detect_image(img)  # yolo检测，返回的是[每个person图像,[四个边界]]
        if wall_image:
            self.changeCoef(wall_image[0])
        nodeList = []  # 存放[{节点坐标},[四个边界]]
        for s_image in r_image:  # 对每个person进行姿态识别，每个lmList里是各节点的坐标
            cap = np.array(s_image[0], dtype=np.uint8)
            lmList = self.detector.findPosition(cap)
            if lmList is not None:
                nodeList.append([lmList, s_image[1]])
        res = self.whetherWarning(nodeList)
        if wall_image:
            res.append([wall_image[0], 5])
        return img, res

    def changeCoef(self, wall_list):
        climbLine = 0.3
        disLength = 1.4
        nearLength = 1.25

        wallThickness = wall_list[2] - wall_list[0]
        self.wallLine = (wall_list[0] + wall_list[2]) // 2
        self.climbTopLine = wall_list[1] * (1 - climbLine) + wall_list[2] * climbLine
        self.distance = wallThickness * disLength
        self.nearDistance = wallThickness * nearLength

    def whetherWarning(self, personList):
        resList = PrimeDic()
        priority = [3, 2, 1, 0]  # 爬墙，接近，交互，普通
        for person in personList:
            # 首先判断是否在爬墙
            if person[1][3] < self.climbTopLine:  # 28节点的y轴坐标
                resList.append(person[1], priority[0])

            temp = person[0].copy()
            ans = self.desnet(person[0])  # 判断是否在交互 [ conf ]
            person[0] = temp

            for i in personList:
                if (self.wallLine - i[1][0]) * (self.wallLine - person[1][0]) < 0:
                    if distance(i[1][0:2], person[1][0:2]) < self.nearDistance:
                        resList.append(person[1], priority[1])
                        resList.append(i[1], priority[1])
                    if ans[0] >= self.confidence:
                        # 判断对侧是否有人， 如果有人，这两个都标为2
                        if distance(i[1][0:2], person[1][0:2]) < self.distance:
                            resList.append(person[1], priority[2])
                            resList.append(i[1], priority[2])
            resList.append(person[1], priority[3])
        return resList.toList()


if __name__ == '__main__':
    predictor = Predictor()
    processor = Processor()
    videoname = "./img/ttes.mp4"
    saveDir = 'video_output001.avi'

    pTime = 0
    mode = 0

    if mode == 0:  # 视频
        cap = cv2.VideoCapture(videoname)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的搞
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(saveDir, fourcc, 20.0, (width, height))
        cc = 0
        endCC = 20
        while cap.isOpened():
            success, img = cap.read()
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if success:
                image = Image.fromarray(np.uint8(img))
                ig, res = predictor.detectImage(image)
                img, pTime = processor.showImage(image, res, pTime, False)
                out.write(img)  # 写入视频
                cc += 1
                if 0 < endCC < cc:
                    break
        cap.release()  # 释放视频
        out.release()
        cv2.destroyAllWindows()  # 释放所有的显示窗口

    elif mode == 1:  # 摄像头
        cap = cv2.VideoCapture(0)
        cc = 0
        while cap.isOpened():
            success, img = cap.read()
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if success:
                image = Image.fromarray(np.uint8(img))
                ig, res = predictor.detectImage(image)
                img, pTime = processor.showImage(image, res, pTime, False)
                cv2.imshow(img)  # 写入视频
                cc += 1
                if cc > 50:
                    break
        cap.release()  # 释放视频
        cv2.destroyAllWindows()  # 释放所有的显示窗口
