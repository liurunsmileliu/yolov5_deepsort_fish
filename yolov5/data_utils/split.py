# encoding: utf-8
"""
@author: Liurunzhi
@time: 2021/7/12 20:50
@Describe: 数据分割
"""
import os
import random
import argparse

# parser = argparse.ArgumentParser()
xml_path = '/home/lrz/Documents/Codes/yolov5-master/VOCData/Annotations'
txt_path = '/home/lrz/Documents/Codes/yolov5-master/VOCData/labels'
# parser.add_argument(xml_path, default='/home/lrz/Documents/Codes/Yolov5-deepsort/VOCData/Annotations', type=str, help='input xml label path')
# parser.add_argument(txt_path, default='/home/lrz/Documents/Codes/Yolov5-deepsort/VOCData/labels', type=str, help='output txt label path')
# opt = parser.parse_args()

trainval_percent = 1.0
train_percent = 0.9
# xmlfilepath = opt.xml_path
xmlfilepath = xml_path
# txtsavepath = opt.txt_path
txtsavepath = txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)  # 总计xml文件数
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()


