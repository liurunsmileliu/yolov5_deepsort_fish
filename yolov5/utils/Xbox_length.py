# encoding: utf-8
"""
@author: Liurunzhi
@time: 2021/8/18 21:10
@Describe:绘制斜框并计算鱼的长度
"""
import cv2
import math
import numpy as np
import time


def drawXbox_calcDis(fish_img, fishBox):
    """
    :param fish_img:鱼的图片
    :param fishBox:YOLOv5检测之后的矩形框坐标值 xyxy
    :return: 每条鱼的四个坐标点和鱼的长度
    """
    img0 = fish_img.copy()  # [data[0][1]:, data[0][0]:data[0][2]]
    img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    # [[[box1],[box2],[box3],[box4],fish_length]] --- [[[643, 1114], [1023, 1114], [1023, 1299], [643, 1299], 380.0]]
    bbox_length = []
    for i in range(0, len(fishBox)):
        t_img = img0[fishBox[i][1]:, fishBox[i][0]:fishBox[i][2]]
        ret, thresh = cv2.threshold(t_img, 65, 255, cv2.THRESH_BINARY)  # 二值化操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CROSS, kernel)
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 轮廓提取
        for c in contours:
            # 每一个目标的左上角坐标和长宽
            x, y, w, h = cv2.boundingRect(c)
            # 画出该矩形
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)  # int(box)
            box1 = (box + [fishBox[i][0], fishBox[i][1]]).tolist()  # 把坐标映射为原图
            fishLen1 = math.pow(box1[1][0] - box1[0][0], 2) + math.pow(box1[1][1] - box1[0][1], 2)
            fishLen2 = math.pow(box1[2][0] - box1[1][0], 2) + math.pow(box1[2][1] - box1[1][1], 2)
            fLen = math.sqrt(max(fishLen1, fishLen2))
            if w > 20:
                box1.append(fLen)
                bbox_length.append(box1)
                # 画出框
                for j in range(0, 3):
                    cv2.line(fish_img, (box1[j]), (box1[j + 1]), (0, 0, 255), 1)
                cv2.line(fish_img, (box1[3]), (box1[0]), (0, 0, 255), 1)
                # box1.clear()
    for k in range(len(fishBox)):  # 标记长度
        # print(k)
        cv2.putText(fish_img, str(bbox_length[k][-1])[:5], (bbox_length[k][3][0] - 5, bbox_length[k][3][1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (51, 139, 226), 1)
    save_path = 'runs/detect/count/'
    cv2.imwrite(save_path + str(time.time()) + '_result.jpg', fish_img)
    return bbox_length
