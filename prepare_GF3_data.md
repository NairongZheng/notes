# GF3_yza

## SAR处理

原始 SAR 图像是 single 类型，要用代码将其进行截断拉伸，再归一化。

```python

"""
    author:nrzheng
    function:阶段拉伸+归一化
    edition:xxx
    date:2021.12.20
"""
import os
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None

img_path = r'E:\try\try'
save_path = r'E:\data\GF3_yza\SAR'

def main():
    """
        主函数
    """
    images = os.listdir(img_path)
    for i, image in enumerate(images):
        img = Image.open(os.path.join(img_path, image))
        img = np.asarray(img)
        img_mean = np.mean(img)
        rate = 3
        img_new = img.copy()
        img_new[img > (rate * img_mean + 1e-7)] = rate * img_mean
        img_new = (img_new / np.max(img_new) + 1e-7) * 255.
        img_new = Image.fromarray(np.uint8(img_new))
        img_new.save(os.path.join(save_path, '{}.tif'.format(i + 1)))

if __name__ == '__main__':
    main()
```

## 标签处理

原本的类别：

```python

"""
    author:nrzheng
    function:处理标签
    edition:xxx
    date:2021.12.20
"""

import os
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

label_path = r'D:\work\tie_JHC_20211101\input_src'
save_path = r'E:\data\GF3_yza\label'
label_mapping = {0:[0,0,255], 1:[139,0,0], 2:[83,134,139], 
                3:[255,0,0], 4:[205,173,0], 5:[0,255,0], 
                6:[0,139,0], 7:[0,139,139], 8:[139,105,20], 
                9:[189,183,107], 10:[178,34,34]}
label_num = len(label_mapping)

label_dic = [{"name":"water", "rgb":[0,0,255]},
            {"name":"bareoil", "rgb":[139,0,0]},
            {"name":"road", "rgb":[83,134,139]},
            {"name":"industry", "rgb":[255,0,0]},
            {"name":"residential", "rgb":[205,173,0]},
            {"name":"vegetation", "rgb":[0,255,0]},
            {"name":"woodland", "rgb":[0,139,0]},
            {"name":"humanbuilt", "rgb":[189,183,107]}, 
            {"name":"other", "rgb":[178, 34, 34]}]
n_labels = len(label_dic)

def get_cmap():
    labels = np.ndarray((n_labels, 3), dtype='uint8')
    for i in range(0, n_labels):
        labels[i] = label_dic[i]['rgb']
    cmap = np.zeros([768], dtype='uint8')
    index = 0
    for i in range(0, n_labels):
        for j in range(0, 3):
            cmap[index] = labels[i][j]
            index += 1
    print('cmap define finished')
    return cmap

def main():
    """
        主函数
    """
    labels = glob.glob(os.path.join(label_path, '*.png'))
    cmap = get_cmap()
    for ii, label in tqdm(enumerate(labels), total=len(labels)):
        lab = Image.open(label)
        lab = np.asarray(lab)
        temp = lab.copy()
        label_mask = np.zeros((lab.shape[0], lab.shape[1]))
        lab_nd = np.zeros([lab.shape[0], lab.shape[1], label_num])
        for k, v in label_mapping.items():
            label_mask[(((temp[:, :, 0] == v[0]) & (temp[:, :, 1] == v[1])) & (temp[:, :, 2] == v[2]))] = int(k)
        for i in range(label_num):
            lab_nd[:, :, i] = np.array(label_mask == i, dtype='uint8')
        new_label = lab_nd[:, :, 0] * 0 + lab_nd[:, :, 1] * 1 + lab_nd[:, :, 2] * 2 + lab_nd[:, :, 3] * 3 + lab_nd[:, :, 4] * 4 + lab_nd[:, :, 5] * 5\
             + lab_nd[:, :, 6] * 6 + lab_nd[:, :, 7] * 8 + lab_nd[:, :, 8] * 5 + lab_nd[:, :, 9] * 7 + lab_nd[:, :, 10] * 8
        new_label = Image.fromarray(np.uint8(new_label))
        new_label.putpalette(cmap)
        new_label.save(os.path.join(save_path, '{}.png'.format(ii + 1)))

if __name__ == '__main__':
    main()
```



## 切图

```python
import argparse
import os
from PIL import Image
import numpy as np
import math
from tqdm import tqdm
import re

Image.MAX_IMAGE_PIXELS = None

label_dic = [{"name":"water", "rgb":[0,0,255]},
            {"name":"bareoil", "rgb":[139,0,0]},
            {"name":"road", "rgb":[83,134,139]},
            {"name":"industry", "rgb":[255,0,0]},
            {"name":"residential", "rgb":[205,173,0]},
            {"name":"vegetation", "rgb":[0,255,0]},
            {"name":"woodland", "rgb":[0,139,0]},
            {"name":"humanbuilt", "rgb":[189,183,107]}, 
            {"name":"other", "rgb":[178, 34, 34]}]
n_labels = len(label_dic)

def get_cmap():
    labels = np.ndarray((n_labels, 3), dtype='uint8')
    for i in range(0, n_labels):
        labels[i] = label_dic[i]['rgb']
    cmap = np.zeros([768], dtype='uint8')
    index = 0
    for i in range(0, n_labels):
        for j in range(0, 3):
            cmap[index] = labels[i][j]
            index += 1
    print('cmap define finished')
    return cmap

def parse_args():
    parser = argparse.ArgumentParser(description='cut_sar_imag')
    parser.add_argument('--sar_path', help='the path of sar images', default=r'E:\data\GF3_yza\aaa')
    parser.add_argument('--lab_path', help='the path of lab images', default=r'E:\data\GF3_yza\bbb')
    parser.add_argument('--save_sar_path', help='the path to save sar images', default=r'E:\data\GF3_yza\img_test')
    parser.add_argument('--save_lab_path', help='the path to save lab images', default=r'E:\data\GF3_yza\lab_test')
    parser.add_argument('--channels', help='the channel of sar images', default=1)
    parser.add_argument('--height', help='the height of new sar images', default=256)
    parser.add_argument('--width', help='the width of new sar images', default=256)
    parser.add_argument('--stride', help='the step to cut new sar images', default=256)
    parser.add_argument('--ext_sar', help='the extension of new sar images', default='.tif')
    parser.add_argument('--ext_lab', help='the extension of new lab images', default='.png')
    args = parser.parse_args()
    return args

def create_filename(path):
    img_ab_names = []
    img_names = []
    filenames = os.listdir(path)
    for filename in filenames:
        img_name = filename.split('.')[0]
        img_names.append(img_name)
        img_ab_name = os.path.join(path,filename)
        img_ab_names.append(img_ab_name)
    
    return img_ab_names,img_names

def save_sar(pic,save_path, small_pic_name):
    small_path = os.path.join(save_path, small_pic_name)
    pic = Image.fromarray(pic)
    pic.save(small_path)

def save_lab(pic,save_path, small_pic_name, cmap):
    small_path = os.path.join(save_path, small_pic_name)
    pic = Image.fromarray(pic)
    pic.putpalette(cmap)
    pic.save(small_path)


def img_slice(sar_path,lab_path,save_sar_path,save_lab_path,weight,height,stride,channels,ext_sar,ext_lab, cmap):
    sar = Image.open(sar_path)
    sar = np.array(sar)
    lab = Image.open(lab_path)
    lab = np.array(lab)
    Height = sar.shape[0]
    Weight = sar.shape[1]
    row = math.floor((Weight - weight) / stride) + 1
    col = math.floor((Height - height) / stride) + 1
    k = 0
    for i in tqdm(range(row), total=row):
        col_start = i*stride
        col_end = i*stride + weight
        for j in range(col):
            row_start = j*stride
            row_end = j*stride + height
            if channels == 3:
                small_sar = sar[row_start:row_end, col_start:col_end, :]
            elif channels == 1:
                small_sar = sar[row_start:row_end, col_start:col_end]
            small_lab = lab[row_start:row_end, col_start:col_end]
            k += 1
            small_name = str(i + 1).rjust(6, '0') + '_row' + str(j + 1).rjust(6, '0') + '_col_' + str(k).rjust(8, '0')
            small_sar_name = 'sar_' + small_name + ext_sar
            save_sar(small_sar,save_sar_path,small_sar_name)
            small_lab_name = 'lab_' + small_name + ext_lab
            save_lab(small_lab,save_lab_path,small_lab_name, cmap)

def main():
    args = parse_args()
    cmap = get_cmap()
    sar_ab_paths, sar_names = create_filename(args.sar_path)
    lab_ab_paths, lab_names = create_filename(args.lab_path)
    
    # sar_ab_paths.sort(key=lambda x: (int(re.split('Intensity_EnLee_|.tif',x)[1])))
    # lab_ab_paths.sort(key=lambda x: (int(re.split('Label_|.png',x)[1])))
    for i in range(len(sar_names)):
        sar_path = sar_ab_paths[i]
        lab_path = lab_ab_paths[i]
        img_slice(sar_path, lab_path, args.save_sar_path, args.save_lab_path, args.height, args.width, 
                    args.stride, args.channels, args.ext_sar, args.ext_lab, cmap)

if __name__ == '__main__':
    main()
```





