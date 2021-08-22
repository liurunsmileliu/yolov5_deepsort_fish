# encoding: utf-8
"""
@author: Liurunzhi
@time: 2021/7/12 20:11
@Describe: 把所有标注文件复制到./VOCData/Annotations;图片复制到./VOCData/images
"""
import gevent
from gevent import monkey

monkey.patch_all()
import os
import shutil
import time


def copy_file(i):
    global num
    num += 1
    print(i,num)
    old_path = r'D:\zjf_workspace\001-地标、利器、服饰\004文本\baidu_isbn5\新建文件夹\txt'
    new_path = r'D:\zjf_workspace\001-地标、利器、服饰\004文本\百度isbn-json-非selenium5'
    name, suffix = i.split('.json')
    name = name.replace('.', '')
    old_name = old_path + '\\' + i
    new_name = new_path + '\\' + name + ".json"
    shutil.copyfile(old_name, new_name)


if __name__ == '__main__':

    start_time = time.time()

    # 需要被复制的文件夹
    old_path = r'D:\zjf_workspace\001-地标、利器、服饰\004文本\baidu_isbn5\新建文件夹\txt'
    all_list = os.listdir(old_path)
    gevent_list = []
    num = 1
    key_num = 0
    for i in all_list:
        key_num += 1
        if key_num >= 1500:
            gevent.joinall(gevent_list)
            gevent.killall(gevent_list)
            gevent_list = []
            key_num = 0
        gevent_list.append(gevent.spawn(copy_file, i))
        # print(i)

    print(len(all_list))
    # print(all_list)
    gevent.joinall(gevent_list)
    end_time = time.time()
    print(end_time - start_time, '秒')

