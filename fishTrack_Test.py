# encoding: utf-8
"""
@author: Liurunzhi
@time: 2021/8/13 15:57
@Describe:测试是否封装完成
"""
import argparse
from fish_track import detect
from yolov5.utils.general import check_img_size
import yaml
import os


cur_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(cur_path, 'deep_sort_pytorch/configs/path_config.yaml')
f = open(yaml_path, 'r', encoding='utf-8')
yaml_con = f.read()
res = yaml.load(yaml_con, Loader=yaml.SafeLoader)

parser = argparse.ArgumentParser()
parser.add_argument('--yolo_weights', type=str,
                    default=res['yolo_weights'],
                    help='model.pt path')
parser.add_argument('--deep_sort_weights', type=str,
                    default=res['deep_sort_weights'],
                    help='ckpt.t7 path')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')  # 0.4
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')  # 0.5
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', type=str, default='2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--evaluate', action='store_true', help='augmented inference')
parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
parser.add_argument('--coordinates', type=list, default=[])
args = parser.parse_args()
args.img_size = check_img_size(args.img_size)

results = detect(args)
for res in results:
    print('坐标值：', res[:, :4])
    print()
