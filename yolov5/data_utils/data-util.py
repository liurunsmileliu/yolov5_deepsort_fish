# encoding: utf-8
"""
@author: Liurunzhi
@time: 2021/7/12 12:35
@Describe: 删除无效文件
"""

import os


def file_name(file_dir):
    jpg_files = []
    xml_files = []
    for root, dirs, files in os.walk(file_dir):
        print('总文件数====》', len(files))
        for file in files:
            # print(file)
            if '.xml' in file:
                # print(file)
                xml_files.append(file[:-4])
        print('xml文件数===》', len(xml_files))
        count = 0
        for file in files:
            if file[:-4] not in xml_files:
                # print('yes')
                os.remove(file_dir + file)
                count += 1
    # print('')
    print('删除文件数===》', count)
    print('删除成功！！')
    # print(root)  # 当前目录路径
    # print(dirs)  # 当前路径下所有子目录
    # print(files)  # 当前路径下所有非目录子文件
    # print(jpg)
    # print(xml_files)


def ReName(file_dir):
    """
    修改文件名
    :param file_dir:文件路径
    :return: 无
    """
    for root, dirs, files in os.walk(file_dir):
        print(files)
        for file in files:
            os.rename(str(file_dir + file), root + file_dir[48:-1] + '_' + str(file))
        print('修改成功！！')


from xml.dom.minidom import parse
import xml.dom.minidom


def modify_xml(xml_path):
    for root, dir, files in os.walk(xml_path):
        # print(len(root))
        for file in files:
            xml_lists = []
            if '.xml' in file:
                xml_lists.append(root+file)
                # print(file)
            for xml_list in xml_lists:
                # print(xml_list)
                print(xml_list[len(root):-4])
                xml = parse(xml_list)
                # print('=====>', xml)
                rootNote = xml.documentElement  # 拿到文档根元素
                # filename = rootNote.getElementsByTagName('filename')[0].toxml()[10:-11]
                # print('需要修改的文件：', filename)
                filename = rootNote.getElementsByTagName('filename')[0]
                folder = rootNote.getElementsByTagName('folder')[0]
                filename.childNodes[0].data = folder.childNodes[0].data + '_' + xml_list[len(root):-4]
                # filename.childNodes[0].data = folder.childNodes[0].data + '_'
                # filename = 'test01_' + filename
                # print('新的命名===》', filename)
                with open(xml_list, 'w') as f:
                    xml.writexml(f, addindent='')
                print('%s -修改成功！！' % xml_list[55:-4])


if __name__ == '__main__':
    file_dir = '/home/lrz/Documents/Codes/Yolov5-deepsort/video/qculus/'
    print(file_dir[48:-1])
    xml_path = '/home/lrz/Documents/Codes/Yolov5-deepsort/video/qculus/'
    # file_name(file_dir)  # 删除无注解文件
    # modify_xml(xml_path)  # 修改xml文件中的filename
    ReName(file_dir)  # 文件重命名
