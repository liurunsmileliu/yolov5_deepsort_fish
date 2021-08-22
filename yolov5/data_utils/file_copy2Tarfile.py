# encoding: utf-8
"""
@author: Liurunzhi
@time: 2021/7/12 20:20
@Describe:
"""
import os
import shutil


def file2TarFile(org_filepath, jpg_file, xml_file):
    for root, dir, files in os.walk(org_filepath):
        # root： /home/lrz/Documents/Codes/Yolov5-deepsort/VOCData/test01/
        print(root)
        for file in files:
            # print(file)  # jpg/xml
            if file[-3:] == 'jpg':
                # print('---------------------------')
                shutil.copy(root+file, jpg_file)
            else:
                # print(file)
                shutil.copy(root+file, xml_file)
        print('复制完成！')


if __name__ == '__main__':
    org_filepath = '/home/lrz/Documents/Codes/Yolov5-deepsort/VOCData/qculus/'
    jpg_file = '/home/lrz/Documents/Codes/Yolov5-deepsort/VOCData/images/'
    xml_file = '/home/lrz/Documents/Codes/Yolov5-deepsort/VOCData/Annotations/'
    file2TarFile(org_filepath, jpg_file, xml_file)
