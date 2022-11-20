import numpy as np
import torch
import torch.nn as nn

from YOLOX.YoloBody import YoloBody
from YOLOX.utils import (cvtColor, preprocess_input, resize_image)
from YOLOX.utils_bbox import decode_outputs, non_max_suppression


class YOLO:
    model_path = 'model_data/yolox_s.pth'
    input_shape = [640, 640]  # 输入图片的大小，必须为32的倍数。
    confidence = 0.5  # 只有得分大于置信度的预测框会被保留下来
    nms_iou = 0.3  # 非极大抑制所用到的nms_iou大小
    letterbox_image = True  # 该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，

    def __init__(self, path='./', use_cuda=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = path + self.model_path
        self.class_names, self.num_classes = (["person", "wall"], 2)
        self.cuda = use_cuda
        self.net = YoloBody(self.num_classes)

        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #   检测图片
    def detect_image(self, image):
        #   获得输入图片的高和宽
        image_shape = np.array(np.shape(image)[0:2])
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = cvtColor(image)
        #   给图像增加灰条，实现不失真的resize
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #   添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        person_image_list = []
        wall_list = []

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #   将图像输入网络当中进行预测！

            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)

            #   将预测框进行堆叠，然后进行非极大抑制
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)

            if results[0] is None:
                return person_image_list, wall_list

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            if predicted_class == 'person':
                box = top_boxes[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
                cropImage = image.crop([left, top, right, bottom])
                person_image_list.append([cropImage, np.array([left, top, right, bottom])])
            elif predicted_class == 'wall':
                box = top_boxes[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
                wall_list.append(np.array([left, top, right, bottom]))

        return person_image_list[:], wall_list[:]
