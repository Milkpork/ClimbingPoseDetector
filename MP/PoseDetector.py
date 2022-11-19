import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageDraw


def rotate(node1, node2, node3):
    tpNode = node2 - node1
    length = np.hypot(tpNode[0], tpNode[1])
    # print(length)
    cosx = (node2[1] - node1[1]) / length
    sinx = (node2[0] - node1[0]) / length
    pp = np.array([[cosx, -sinx], [sinx, cosx]])
    res = np.dot(pp, node3).reshape(1, 2)
    return res


def normalization(dic):
    if len(dic) == 6:
        value = np.array(list(dic.values()))
        value = np.round(value, 2)
        return value
    elif len(dic) >= 0:
        raise ValueError("not enough 6 parameters")


class PoseDetector:
    need_node = [12, 14, 16, 24, 26, 28]

    def __init__(self, mode=False, model_complexity=1, smooth=True, enable_segmentation=False,
                 smooth_segmentation=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth, self.enable_segmentation,
                                     self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPosition(self, img):
        """
            返回[, 6]的节点坐标，未进行归一化
        """
        lmList = {}
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            for index, lm, in enumerate(results.pose_landmarks.landmark):
                h, w, c = imgRGB.shape
                if index in self.need_node:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList[index] = np.array([cx, cy])
        try:
            lmList = normalization(lmList)
        except ValueError:
            lmList = None
        return lmList
