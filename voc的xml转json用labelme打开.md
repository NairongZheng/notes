<font color='green' size=4> edited by nrzheng，2022.2.21</font>

# VOC数据集简略介绍

## 1. Pascal VOC（Pascal Visual Object Classes）

VOC数据集是目标检测经常用的一个数据集，从05年到12年都会举办比赛（任务有： Classification 、<font color='red'>Detection </font>、 Segmentation、Person Layout）

## 2. Annotations

标签文件夹是Annotations，采用<font color='red'>xml</font>格式存储（以下以SAR目标检测数据集SSDD为例介绍各字段）

每个xml内容大致如下：

```xml
<annotation verified="no">
  <folder>JPEGImages</folder>
  <filename>000001</filename>
  <path>E:\\data\\detection\\SSDD\\JPEGImages\\000001.jpg</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>416</width>
    <height>323</height>
    <depth>1</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>ship</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>208</xmin>
      <ymin>50</ymin>
      <xmax>273</xmax>
      <ymax>151</ymax>
    </bndbox>
  </object>
</annotation>

```

- folder：该标签文件对应的图像的文件夹
- filename：文件名（VOC中放的是加后缀的图像名）
- path：对应图像的绝对路径（VOC中这部分内容比较详细，source也比较详细）
- size：图像的尺寸，包括宽、高、通道数
- segmented：是否用于分割，0为否
- object：每个object都是一个目标，有的图像中会有多个目标就对应多个object
  - name：目标的类别 / 名称
  - pose：摄像头角度
  - truncated：是否被截断，0表示完整
  - difficult：目标是否难以识别，0表示容易识别
  - bndbox：目标的位置，由于都是正框，所以这里给出的4个值是xmin、ymin、xmax、ymax

## 3. SSDD数据介绍

### 3.1. 文件夹结构

```
SSDD
   ├─Annotations
   ├─JPEGImages
```

- Annotations：标签文件，xml格式
- JPEGImages：图像文件，jpg格式
- （基本都是这种）

### 3.2. 标签及图像

Annotations和JPEGImages文件夹的内容如下所示：

![](https://cdn.jsdelivr.net/gh/Damon-X46/ImgHosting/images/20220221_标签及图像展示.jpg)

一般在使用之前，要检查以下数据是否准确（个人习惯），但是目标检测的数据集在原图上都没办法看到标签打的情况，因为标签都是独立的一个文件，跟语义分割的标签直接可视化不一样，目标检测的标签都是一些文本。

所以，为了检查标签，需要把标签文件到原图上可视化。这里借助的是工具<font color='red'>labelme</font>

# labelme介绍

## 1. labelme

Labelme 是一个图形界面的图像标注软件。其的设计灵感来自于[这里](http://labelme.csail.mit.edu/) 。它是用 Python 语言编写的，图形界面使用的是 Qt（PyQt）。

labelme下载：[点击这里](https://pan.baidu.com/s/1YAajEfknEiznmUJLociNug)，提取码：2o1k

## 2. labelme使用及界面展示

- 一般都是直接打开文件夹，把图像文件加载进来
- 一般都会把file---Save With Image Data关掉。因为这个打开的话，标签文件会保存原图的信息，很长很长，没有必要
- 这样就可以开始标注了，可以file---Change Output Dir选择标签输出的文件夹
  - 若已有标签，且标签跟原图在一个文件夹，会自动读取标签显示框框
  - 若已有标签，但标签跟原图不在一个文件夹，file---Change Output Dir选到标签所在文件夹，就会自动读取
  - 界面右下角有图像的绝对路径，前面带勾的表示该图像有标签

labelme的界面如下：

![](https://cdn.jsdelivr.net/gh/Damon-X46/ImgHosting/images/20220221_labelme界面展示.jpg)

## 3. labelme标签格式

labelme采用json文件格式存储标签，如下：

```json
{
  "version": "4.5.6",
  "flags": {},
  "shapes": [
    {
      "label": "ship",
      "points": [
        [
          202.66666666666669,
          58.333333333333336
        ],
        [
          229.33333333333331,
          148.33333333333334
        ],
        [
          282.3333333333333,
          134.66666666666666
        ],
        [
          243.66666666666669,
          42.0
        ]
      ],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "000001.jpg",
  "imageData": null,
  "imageHeight": 323,
  "imageWidth": 416
}
```

- flags：描述你分类的标签有哪些，比如{0, 1}表示二分类，或者多分类也可以{water, road, industry, ...}。一般都是空的
- shapes：里面就是标注的信息，多个的话则shapes里就有多个”{}“
  - label：类别
  - points：构成这个目标的所有点
  - shape_type：形状的类型。polygon代表多边形
  - 因为labelme不局限于画矩形正框，可以有好多点表示一个目标，所以points里面是一个个的点，跟VOC数据用xy的最大最小坐标表示不一样
- imagePath：图像的路径（我喜欢用绝对路径）
- imageData：图像的一些信息。上面讲的取消就会输出null，放出来就是一长串字符
- imageHeight：图像的高
- imageWidth：图像的宽

# VOC的xml转json

前面的介绍知道，要用labelme可视化VOC数据，就要把VOC数据集中的标签（xml格式）转成labelme可识别的标签（json格式）。只要json中的字段对应的上，转换之后就可以打开。转换后可视化如下：

![](https://cdn.jsdelivr.net/gh/Damon-X46/ImgHosting/images/20220221_转换后展示.jpg)

可以看到，数据集标签没有问题。

代码如下：

```python
"""
    Author:DamonZheng
    Function:xml2json(for labelme)
    Edition:1.0
    Date:2022.2.21
"""

import argparse
import glob
import os
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

def parse_args():
    """
        参数配置
    """
    parser = argparse.ArgumentParser(description='xml2json')
    parser.add_argument('--raw_label_dir', help='the path of raw label', default=r'E:\data\detection\ship_detection_online\Annotations_new')
    parser.add_argument('--pic_dir', help='the path of picture', default=r'E:\data\detection\ship_detection_online\JPEGImages')
    parser.add_argument('--save_dir', help='the path of new label', default=r'E:\data\detection\ship_detection_online\label_json')
    args = parser.parse_args()
    return args

def read_xml_gtbox_and_label(xml_path):
    """
        读取xml内容
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)
    points = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        pose = obj.find('pose').text
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        box = [xmin, ymin, xmax, ymax]
        point = [cls, box]
        points.append(point)
    return points, width, height

def main():
    """
        主函数
    """
    args = parse_args()
    labels = glob.glob(args.raw_label_dir + '/*.xml')
    for i, label_abs in tqdm(enumerate(labels), total=len(labels)):
        _, label = os.path.split(label_abs)
        label_name = label.rstrip('.xml')
        img_path = os.path.join(args.pic_dir, label_name + '.jpg')
        points, width, height = read_xml_gtbox_and_label(label_abs)
        json_str = {}
        json_str['version'] = '4.5.6'
        json_str['flags'] = {}
        shapes = []
        for i in range(len(points)):
            shape = {}
            shape['label'] = points[i][0]
            shape['points'] = [[points[i][1][0], points[i][1][1]], 
                                [points[i][1][0], points[i][1][3]], 
                                [points[i][1][2], points[i][1][3]],
                                [points[i][1][2], points[i][1][1]]]
            shape['group_id'] = None
            shape['shape_type'] = 'polygon'
            shape['flags'] = {}
            shapes.append(shape)
        json_str['shapes'] = shapes
        json_str['imagePath'] = img_path
        json_str['imageData'] = None
        json_str['imageHeight'] = height
        json_str['imageWidth'] = width
        with open(os.path.join(args.save_dir, label_name + '.json'), 'w') as f:
            json.dump(json_str, f, indent=2)

if __name__ == '__main__':
    main()
```







