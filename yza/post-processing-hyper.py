"""
    Author:damonzheng
    Function:post-processing-hyper
    Edition:1.0
    Date:2023.1.27
"""

import os
import argparse
import numpy as np
import glob
from scipy.io import loadmat
from PIL import Image
import cv2

# # HSN/S3ER:
# row_rgb = {0:[219,94,86], 1:[219,183,86], 2:[167,219,86], 
#             3:[86,219,94], 4:[86,219,183], 5:[86,167,219], 
#             6:[94,86,219], 7:[183,86,219], 8:[219,86,167]}

# # HSI-BERT/RSSAN/SSRN/DAGCN
# row_rgb = {0:[255,0,0], 1:[0,255,0], 2:[0,0,255], 
#             3:[255,0,255], 4:[102,102,0], 5:[205,149,12], 
#             6:[255,255,0], 7:[139,134,130], 8:[0,255,255]}

# SSTFF/ASSMN
row_rgb = {0:[0,0,255], 1:[0,255,0], 2:[0,255,255], 
            3:[0,127,0], 4:[255,0,255], 5:[165,82,41], 
            6:[127,0,127], 7:[255,0,0], 8:[255,255,0]}

new_rgb = {0:[128,138,135], 1:[0,255,0], 2:[0,255,255], 
            3:[34,139,34], 4:[255,0,255], 5:[199,97,20], 
            6:[160,32,240], 7:[255,0,0], 8:[255,255,0]}


def parse_args():
    parser = argparse.ArgumentParser(description='post-processing-hyper')
    parser.add_argument('--lab_path', help='the path of label', default=r'C:\Users\95619\Desktop\PU\ASSMN.png')
    parser.add_argument('--gt_path', help='the path of ground truth', default=r'C:\Users\95619\Desktop\PU\paviaU_gt.mat')
    parser.add_argument('--threshold', help='the threshold to change RGB to standard', default = 5)
    parser.add_argument('--save_path', help='the path of save file', default=r'C:\Users\95619\Desktop\PU\after')
    args = parser.parse_args()
    return args


def find_nearest(args, a, a0):
    new_a = a.copy()
    dis = a - a0
    dis = np.abs(dis)
    dis = np.sum(dis, -1)
    pos = np.where(dis < args.threshold)
    new_a[pos] = a0
    return new_a


def three2one(lab):
    temp = lab.copy()
    label_mask = np.zeros((lab.shape[0], lab.shape[1]))
    for i, (k, v) in enumerate(row_rgb.items()):
        label_mask[(((temp[:, :, 0] == v[0]) & (temp[:, :, 1] == v[1])) & (temp[:, :, 2] == v[2]))] = int(k)
    return label_mask


def one2three(args, lab):
    new_label = np.zeros([args.h, args.w, 3])
    for i, (k, v) in enumerate(new_rgb.items()):
        locals()['cls' + str(k)] = np.where(lab == k)
        new_label[eval('cls' + str(k))] = v
    return new_label


def main():
    args = parse_args()
    # labels = glob.glob(os.path.join(args.lab_path, '*.png'))
    # 打开gt
    gt_dict = loadmat(args.gt_path)
    gt = gt_dict['paviaU_gt']
    gt_h, gt_w = gt.shape
    # args.h, args.w = h, w

    # 打开要处理的lab
    lab_pth = args.lab_path
    lab = Image.open(lab_pth)
    lab = lab.convert('RGB')
    w, h = lab.size
    args.h, args.w = h, w

    # resize
    # lab = lab.resize((w, h))
    lab = np.asarray(lab)

    # lab = find_nearest(args, lab, row_rgb[3])
    # 有的RGB不标准，总差1, 2，替换成标准的
    for k, v in row_rgb.items():
        lab = find_nearest(args, lab, row_rgb[k])
    
    # 转成单通道
    lab_one = three2one(lab)

    # 转成三通道
    lab_three = one2three(args, lab_one)

    # 黑色还原
    # black = np.where(gt == 0)
    # lab_three[black] = [0, 0, 0]

    # resize+黑色还原
    aaa = Image.fromarray(np.uint8(lab_three))
    aaa = aaa.resize((gt_w, gt_h))
    aaa = np.asarray(aaa)
    black = np.where(gt == 0)
    bbb = aaa.copy()
    bbb[black] = [0, 0, 0]

    # 保存
    bbb = Image.fromarray(np.uint8(bbb))
    bbb.save(os.path.join(args.save_path, os.path.split(lab_pth)[1]))


if __name__ == '__main__':
    main()
