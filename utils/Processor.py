from PIL import ImageDraw
import numpy as np
import cv2
import time

from .Alarm import Alarm


class Processor:
    def __init__(self):
        self.alarm = Alarm()

    def showImage(self, img, boxList, pTime=None, isImg=False):
        self.detectAlarm(boxList)
        thickness = 5  # 线的长度
        draw = ImageDraw.Draw(img)
        # 黑，红，绿，蓝，未使用，黄
        # 0为正常，1为节点判断的交互，2为距离判断的交互，3为爬墙，5为墙体
        colorTuple = ((0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255))
        for box in boxList:
            for i in range(thickness):
                draw.rectangle([box[0][0] + i, box[0][1] + i, box[0][2] - i, box[0][3] - i], outline=colorTuple[box[1]])
        del draw
        cTime = time.time()
        img = np.asarray(img)
        if isImg:
            cv2.putText(img, "1", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            cv2.namedWindow('Image', 0)
            return img
        else:
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            return img, pTime

    def detectAlarm(self, ls):
        pls = [i[1] for i in ls if i[1] in range(0, 5)]
        if len(pls) == 0:
            self.alarm.clear()
            return
        if max(pls) == 3:
            self.alarm.climbAlarm()
        elif max(pls) in [2, 1]:
            self.alarm.touchAlarm()
        else:
            self.alarm.clear()
