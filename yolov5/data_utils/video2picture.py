# encoding: utf-8
"""
@author: Liurunzhi
@time: 2021/7/12 9:21
@Describe: 把视频转为图片
"""
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize


def Video2Pic():
    videoPath = "/home/lrz/Documents/Codes/Yolov5-deepsort/video/131744.mp4"  # 读取视频路径
    imgPath = "/home/lrz/Documents/Codes/Yolov5-deepsort/video/131744/"  # 保存图片路径

    cap = cv2.VideoCapture(videoPath)
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    print(suc)
    frame_count = 0
    img_count = 0
    while (suc):
        frame_count += 1
        suc, frame = cap.read()
        # if(frame_count % 2 == 0):
        print(frame)
        # cv2.imwrite(imgPath + str(frame_count).zfill(5), frame)
        cv2.imwrite(imgPath+'fish_'+str(frame_count)+'.jpg', frame)
        img_count += 1
        cv2.waitKey(1)
    print(img_count)
    cap.release()
    print("视频转图片结束！")


if __name__ == '__main__':
    Video2Pic()
