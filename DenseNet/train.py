import time

import numpy as np
from torch.optim import SGD
import torch
import torch.nn as nn

from DenseNet import DenseNet

use_past_model = True
annoDir = "./Annotations/annotation_"
logDir = "./Annotations/log.txt"
inputNum = [71, 74]  # 0,1 的数量
use_gpu = torch.cuda.is_available()
print("gpu is available:", use_gpu)

modelPath = '../' if use_past_model is True else './'


def getData(inputNum, annoDir):
    """load data"""
    # 获取数据大小用于判断
    with open(logDir, 'r') as f:
        classCount = list(map(int, f.read().split()))
    for i in range(2):
        if inputNum[i] > classCount[i + 1]:
            print("inputNumber is too big!")
            inputNum[i] = classCount[i + 1]

    '''采集随机数据'''
    with open(annoDir + "0.txt", 'r') as f:
        normalData = f.read().split('\n')[:-1]
    normalData = np.array(normalData)
    normalData = np.random.choice(a=normalData, size=inputNum[0], replace=False, p=None)

    with open(annoDir + "1.txt", 'r') as f:
        transmitData = f.read().split('\n')[:-1]
    transmitData = np.array(transmitData)
    transmitData = np.random.choice(a=transmitData, size=inputNum[1], replace=False, p=None)

    allList = np.concatenate([normalData, transmitData], 0)
    np.random.shuffle(allList)

    xList = []
    yList = []
    for data in allList:
        tempX, tempY = data.rsplit(" ", 1)
        temp = [0]
        temp[0] = eval(tempY)
        xList.append(eval(tempX))
        yList.append(temp)
    # xList = torch.Tensor(xList)
    xList = np.array(xList)
    yList = np.array(yList)
    # yList = torch.Tensor(yList)
    return xList, yList


def getData2(path='./res.txt'):
    with open(path, 'r') as f:
        allList = f.read().split('\n')
    np.random.shuffle(allList)
    xList = []
    yList = []
    for data in allList:
        # print(data)
        tempX, tempY = data.rsplit(" ", 1)
        temp = [0]
        temp[0] = eval(tempY)
        xList.append(eval(tempX))
        yList.append(temp)
    xList = np.array(xList)
    yList = np.array(yList)
    return xList, yList


def devideDataSet(xList, yList, pp=None):
    if pp is None:
        pp = [0.2, 0.8]
    lenX, lenY = xList.shape[0], yList.shape[0]
    # print(lenX[0])
    if lenX != lenY:
        print(lenX, lenY)
        raise ValueError('fuck it')
    rangeList = np.array(range(0, lenY))
    np.random.shuffle(rangeList)
    spPonit = int(pp[0] * lenY)
    print(spPonit)
    testIndexList, trainIndexList = np.split(rangeList, [spPonit])

    # testList
    testListX = torch.tensor(xList[testIndexList], dtype=torch.float32)
    testListY = torch.tensor(yList[testIndexList], dtype=torch.float32)
    trainListX = torch.tensor(xList[trainIndexList], dtype=torch.float32)
    trainListY = torch.tensor(yList[trainIndexList], dtype=torch.float32)

    return testListX, testListY, trainListX, trainListY


'''训练'''
if use_gpu:
    myNet = DenseNet(modelPath).cuda()
    loss_fn = nn.MSELoss().cuda()
    tensor_cv = torch.Tensor().cuda()
else:
    myNet = DenseNet(modelPath)
    loss_fn = nn.MSELoss()
optimizer = SGD(myNet.parameters(), 0.0001)
epoch = 1000
t = time.time()
# xList, yList = getData(inputNum, annoDir)
xList, yList = getData2()
testListX, testListY, trainListX, trainListY = devideDataSet(xList, yList)
lens = trainListY.shape[0]
testLens = testListY.shape[0]
for i in range(epoch + 1):
    loss = None
    for j in range(lens):
        y_predict = myNet(trainListX[j])
        # print("shape: ", y_predict, trainListY[j])
        loss = loss_fn(y_predict, trainListY[j])
        optimizer.zero_grad()  # 用优化器类将梯度设置为0
        loss.backward()
        optimizer.step()
    if i % 50 == 0:
        conf = 0.5
        print(f"{i}/{epoch}", "loss : ", loss.item(), "  time: ", time.time() - t)
        corrCount = 0
        allCount = 0
        for j in range(testLens):
            y_predict = myNet(testListX[j])
            tempIndex = torch.argmax(y_predict)
            tempIndex2 = torch.argmax(testListY)
            if tempIndex == tempIndex2 and y_predict[tempIndex] > conf:
                corrCount += 1
            allCount += 1
        print("accuracy : ", corrCount / allCount * 100, "%")
        torch.save(myNet.state_dict(), "model_data.pth")  # 保存参数
torch.save(myNet.state_dict(), "model_data.pth")  # 保存参数
print("train finished")
