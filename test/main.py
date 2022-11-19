import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture('./testFold/ttes.mp4')
    # 视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频的宽
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频的搞
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
    fourcc = fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 视频对象的输出
    out = cv2.VideoWriter('video_output001.avi', fourcc, 20.0, (width, height))
    # out = cv2.VideoWriter('out.avi', fourcc, 20.0, (width, height))
    cc = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # cv2.imshow('fame', frame)
        key = cv2.waitKey(1)
        out.write(frame)  # 写入视频
        print("write success")
        cc += 1
        if cc > 50:
            break
        if key == ord('q'):
            break
    cap.release()  # 释放视频
    out.release()
    cv2.destroyAllWindows()  # 释放所有的显示窗口
