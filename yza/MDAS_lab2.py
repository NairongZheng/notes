"""
    Author:damonzheng
    Function:MDAS_lab
    Edition:1.0
    Date:2023.2.3
"""

import os
import argparse
import numpy as np
import glob
from scipy.io import loadmat
from PIL import Image
import cv2


row_rgb = {0: [255, 235, 205], 1: [0, 0, 255], 2: [255, 0, 0],
           3: [64, 224, 208], 4: [160, 32, 240], 5: [25, 25, 112],
           6: [176, 224, 230], 7: [112, 128, 105], 8: [0, 255, 0],
           9: [199, 97, 20], 10: [61, 89, 171], 11: [255, 97, 0],
           12: [48, 128, 20], 13: [84, 38, 18], 14: [0, 255, 255], 
           15: [255, 255, 0], 16:[218,165,105], 17:[163,148,128], 
           18:[0,199,140], 19:[135,38,87]}


def parse_args():
    parser = argparse.ArgumentParser(description='post-processing-hyper')
    parser.add_argument('--data_path', help='the path of data', default=r'D:\code_matlab\play_play')
    parser.add_argument('--save_path', help='the path of save file', default=r'D:\code_matlab\play_play\labels.png')
    args = parser.parse_args()
    return args


def one2three(h, w, lab):
    new_label = np.zeros([h, w, 3])
    for i, (k, v) in enumerate(row_rgb.items()):
        locals()['cls' + str(k)] = np.where(lab == k)
        new_label[eval('cls' + str(k))] = v
    return new_label


def func(output_1, index, obj):
    obj_cls = np.unique(obj)
    for i in range(len(obj_cls)):
        num = obj_cls[i]
        if num != 0:
            pos = np.where(obj == num)
            output_1[pos] = index
            index += 1
    return output_1, index


def main():
    args = parse_args()
    # 读数据
    buildings = loadmat(os.path.join(args.data_path, 'buildings.mat'))[
        'buildings']
    water = loadmat(os.path.join(args.data_path, 'water.mat'))['water']
    landuse = loadmat(os.path.join(args.data_path, 'landuse.mat'))['landuse']
    h, w = landuse.shape

    landuse_cls = np.unique(landuse)
    water_cls = np.unique(water)
    buildings_cls = np.unique(buildings)

    # 把landuse重新从0开始编
    output_1 = np.zeros([h, w])
    index = 0
    for i in range(len(landuse_cls)):
        num = landuse_cls[i]
        pos = np.where(landuse == num)
        output_1[pos] = i
        index += 1

    # 把water和buildings贴到landuse
    output_1, index = func(output_1, index, water)
    output_1, index = func(output_1, index, buildings)

    new_lab = one2three(h, w, output_1)
    aaa = Image.fromarray(np.uint8(new_lab))
    aaa.save(args.save_path)
    pass


if __name__ == '__main__':
    main()
