# encoding: utf-8
"""
@author: Liurunzhi
@time: 2021/7/13 21:43
@Describe:
"""
import cv2

fname = '/home/lrz/Documents/Codes/yolov5-master/data/images/img_fish_420.jpg'
img = cv2.imread(fname)
# 画矩形框 距离靠左靠上的位置
pt1 = (285, 219)  # 左边，上边   #数1 ， 数2
pt2 = (317, 237)  # 右边，下边  #数1+数3，数2+数4
result = cv2.rectangle(img, pt1, pt2, (0, 255, 255), 1)
print(result)
# cv2.imshow('show', result)

# a = 'people'  # 类别名称
# b = 0.596  # 置信度
# font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
# imgzi = cv2.putText(img, '{} {:.3f}'.format(a, b), (651, 460 - 15), font, 1, (0, 255, 255), 4)
# 图像，      文字内容，      坐标(右上角坐标)，字体， 大小，  颜色，    字体厚度
cv2.imwrite('/home/lrz/Documents/Codes/yolov5-master/data/images/img_fish_420_1.jpg', result)
# cv2.imshow(result)
print('yes!!!')