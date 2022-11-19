"""
  Opencv:是一个跨平台的计算机视觉库，可以提供Python接口，以实现图像处理和计算机
视觉方面的很多算法
"""
import cv2

video_path = 'E:\\python\\pycharm\\ClimbingPoseDetector\\img\\vvv.mp4'  # 视频地址
output_path = 'E:\\python\\pycharm\\ClimbingPoseDetector\\img\\pic7\\'  # 输出文件夹
interval = 5  # 每间隔10帧取一张图片

if __name__ == '__main__':
    num = 301
    vid = cv2.VideoCapture(video_path)  # 读取视频
    while vid.isOpened():  # 判断视频对象是否成功读取，成功读取视频对象返回True
        is_read, frame = vid.read()
        # 按帧读取视频，is_read为布尔型，正确读取则返回True，否则返回False
        # frame为每一帧的图像，读取的图像为RGB模式
        if is_read:
            if num % interval == 1:
                file_name = num // interval
                frame = cv2.flip(frame, -1)
                cv2.imwrite(output_path + str(file_name) + '.jpg', frame)
                # imwrite(filename,image)  filename:代表文件名的字符串。文件名必须包含图像格式
                # image:就是要保存的图像。
                # 111.jpg 代表第111帧
            num += 1

        else:
            break
