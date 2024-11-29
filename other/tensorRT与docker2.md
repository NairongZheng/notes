## 镜像

**建议直接看提醒事项！！！**

Dockerfile如下：

```shell
# BASE IMAGE
FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu16.04

SHELL ["/bin/bash","-c"]

WORKDIR /tmp
# copy安装文件
COPY Python-3.6.9.tar.xz /tmp
# 设置 root 密码
RUN echo 'root:password' | chpasswd \
# 安装openssh-server 并配置
  && apt-get update && apt-get -y install openssh-server \
  && sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config \ 
  && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config \
  && mkdir /var/run/sshd \
# 安装python依赖包
  && apt-get -y install build-essential python-dev python-setuptools python-pip python-smbus \
  && apt-get -y install build-essential libncursesw5-dev libgdbm-dev libc6-dev \
  && apt-get -y install zlib1g-dev libsqlite3-dev tk-dev \
  && apt-get -y install libssl-dev openssl \
  && apt-get -y install libffi-dev \
# 安装python 3.6.9
  && mkdir -p /usr/local/python3.6 \
  && tar xvf Python-3.6.9.tar.xz \
  && cd Python-3.6.9 \
  && ./configure --prefix=/usr/local/python3.6 \
  && make altinstall \
# 建立软链接
  && ln -snf /usr/local/python3.6/bin/python3.6 /usr/bin/python \
  && ln -snf /usr/local/python3.6/bin/pip3.6 /usr/bin/pip\
# 清理copy的安装文件
  && apt-get clean \
  && rm -rf /tmp/* /var/tmp/*

EXPOSE 22

CMD ["/bin/bash"]
```

**运行命令**：

```shell
docker build -t + 镜像的名字:版本 .
docker build -t znr_hb_1604_102:v1 .
```

（不用管，没遇到）少了`RUN rm /etc/apt/sources.list.d/cuda.list`的话可能会报如下错误：

```shell
InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
```

（不用管，没遇到）少了`ENV DEBIAN_FRONTEND=noninteractive`的话可能会卡在：

```shell
Configuring tzdata
    ------------------
    Please select the geographic area in which you live. Subsequent configuration
    questions will narrow this down by presenting a list of cities, representing
    the time zones in which they are located.
     1. Africa      4. Australia  7. Atlantic  10. Pacific  13. Etc
     2. America     5. Arctic     8. Europe    11. SystemV
     3. Antarctica  6. Asia       9. Indian    12. US
    Geographic area:
```

创建成功：

```shell
Looking in links: /tmp/tmpqyvxobyp
Collecting setuptools
Collecting pip
Installing collected packages: setuptools, pip
Successfully installed pip-18.1 setuptools-40.6.2
Removing intermediate container d37a55cd4623
 ---> 4049544084be
Step 6/7 : EXPOSE 22
 ---> Running in 58024c221e89
Removing intermediate container 58024c221e89
 ---> da01b7928202
Step 7/7 : CMD ["/bin/bash"]
 ---> Running in fcb885363e11
Removing intermediate container fcb885363e11
 ---> 16ff4f40d217
Successfully built 16ff4f40d217
Successfully tagged znr_hb_1604_102:v1
```



## 容器

### 运行容器

**第一次进容器命令**：

```shell
docker run -it --gpus all --name 容器名字 镜像名字:镜像版本 /bin/bash
docker run -it --gpus all --name znr_hb_yyds znr_hb_1604_102:v1 /bin/bash
```

可以查看看当前cuda版本：

```shell
root@4673f95905cf:/tmp# cat /usr/local/cuda/version.txt
CUDA Version 10.2.89
root@4673f95905cf:/tmp# 
```

### 安装anaconda

安装anaconda：

```bash
bash Anaconda3-2020.11-Linux-x86_64.sh
```

接受license之后有一个询问是否初始话anaconda，默认的是no，记得不要按太快了，要选yes，然后更新一下环境变量：

```shell
source ~/.bashrc
```

然后安装pytorch：

```shell
conda create -n torch python=3.6
conda activate torch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
```

可以用pytorch查看cuda能不能用，还有cudnn版本：

```python
>>> import torch
>>> torch.cuda.is_available()
True
>>> print(torch.backends.cudnn.version())
7605
>>> 
```

要注意，这个版本对后续安装tensorrt很重要！

### 安装tensorRT

到[nvidia官网](https://developer.nvidia.com/nvidia-tensorrt-8x-download)下载，选择跟自己cuda对应版本的。传入需要的文件：

```shell
docker cp 本地文件绝对路径 docker路径
docker cp /home/dbcloud/znr/TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz 4673f95905cf:/
```

解压：

```shell
tar -xvzf TensorRT-7.1.3.4.Ubuntu-16.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
```

export（这步可以跳过，下面会直接添加）：

```shell
export TRT_RELEASE=`pwd`/TensorRT-7.1.3.4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_RELEASE/lib
```

安装python（要注意python版本对应）：

```shell
cd TensorRT-7.1.3.4/python
pip install tensorrt-7.1.3.4-cp36-none-linux_x86_64.whl
```

可以到python中看看能不能import：

```
>>> import tensorrt
>>> tensorrt.__version__
'7.1.3.4'
>>> 
```

有时候会出现问题：

`ImportError: libnvinfer.so.7: cannot open shared object file: No such file or directory`

改.bashrc，再source一下就可以了。[添加环境变量大法](https://www.cnblogs.com/poloyy/p/12187148.html)：

```shell
vim ~/.bashrc
export LD_LIBRARY_PATH=/TensorRT-7.1.3.4/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/TensorRT-7.1.3.4/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
source ~/.bashrc
cd /
```

### 安装torch2trt

torch模型转tensorRT有两种方法：

- torch---onnx---tensorrt
- torch---tensorrt

这边用第二种方法直接转。要用github的一个项目：[torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)。（建议用下面gcunhase的那个！）

```shell
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

有的版本这个不行，也可以试试别的：

```shell
git clone https://github.com/gcunhase/torch2trt.git
git clone https://gitcode.net/mirrors/nvidia-ai-iot/torch2trt.git
```

安装结果：

```shell
byte-compiling build/bdist.linux-x86_64/egg/torch2trt/__init__.py to __init__.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torch2trt/module_test.py to module_test.cpython-36.pyc
byte-compiling build/bdist.linux-x86_64/egg/torch2trt/flatten_module_test.py to flatten_module_test.cpython-36.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying torch2trt.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying torch2trt.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying torch2trt.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying torch2trt.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
zip_safe flag not set; analyzing archive contents...
torch2trt.contrib.qat.layers.__pycache__._utils.cpython-36: module MAY be using inspect.stack
creating dist
creating 'dist/torch2trt-0.4.0-py3.6.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing torch2trt-0.4.0-py3.6.egg
creating /root/anaconda3/envs/torch/lib/python3.6/site-packages/torch2trt-0.4.0-py3.6.egg
Extracting torch2trt-0.4.0-py3.6.egg to /root/anaconda3/envs/torch/lib/python3.6/site-packages
/root/anaconda3/envs/torch/lib/python3.6/site-packages/torch2trt-0.4.0-py3.6.egg/torch2trt/dataset.py:61: SyntaxWarning: assertion is always true, perhaps remove parentheses?
  assert(len(self) > 0, 'Cannot create default flattener without input data.')
Adding torch2trt 0.4.0 to easy-install.pth file

Installed /root/anaconda3/envs/torch/lib/python3.6/site-packages/torch2trt-0.4.0-py3.6.egg
Processing dependencies for torch2trt==0.4.0
Finished processing dependencies for torch2trt==0.4.0
```

安装完可以看到，pip list中已经有了torch2trt：

```shell
Package           Version
----------------- ---------
certifi           2021.5.30
dataclasses       0.8
numpy             1.19.5
packaging         21.3
Pillow            8.4.0
pip               21.2.2
pyparsing         3.0.9
setuptools        58.0.4
tensorrt          7.1.3.4
torch             1.8.0
torch2trt         0.4.0
torchaudio        0.8.0
torchvision       0.9.0
typing_extensions 4.1.1
wheel             0.37.1
```

进入python，试试能否import：

```shell
>>> import torch
>>> import tensorrt
>>> from torch2trt import torch2trt, TRTModule
>>> 
```

### 转模型

接下来就是转模型。

可以先用测试代码试试：

```python
import torch
from torch2trt import torch2trt
from torchvision.models.alexnet import alexnet

# create some regular pytorch model...
model = alexnet(pretrained=True).eval().cuda()

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

# pdb.set_trace()
# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
print('complete')
y = model(x)
y_trt = model_trt(x)
print(torch.max(torch.abs(y-y_trt)))
```

结果如下：

```shell
complete
tensor(1.7881e-06, device='cuda:0', grad_fn=<MaxBackward1>)
```



成功的话再在自己的代码试，代码如下：

```python
import argparse
import os, sys
# import seg_hrnet
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from default import _C as config
from default import update_config
import hrnet
import torch
from torch2trt import TRTModule, torch2trt
from torch.autograd import Variable

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--pretrained', default='/remote-home/znr/xingtubei/code/hrnet/output',
                        type=str, metavar='PATH',
                        help='use pre-trained model path')
    parser.add_argument('--weights_name', default='HRNetW48_epoch_204_fwiou_0.98278_OA_0.9912_docker.pth')
    parser.add_argument('--cfg', default='/remote-home/znr/xingtubei/code/hrnet/experimentals_yaml/juesai/20230302_01.yaml')
    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    # 转成docker提交用的
    weights_name = args.weights_name
    model_state_file = os.path.join(args.pretrained, weights_name)
    pretrained_dict = torch.load(model_state_file)
    name = os.path.splitext(weights_name)[0] + '_docker.pth'
    torch.save(pretrained_dict, os.path.join(args.pretrained, name), _use_new_zipfile_serialization=False)

    # 直接转tensorrt
    model = eval('hrnet.get_seg_model')(config)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(os.path.join(args.pretrained, name))
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval().cuda()
    x = torch.ones((1, 3, 512, 512)).cuda()
    model_trt = torch2trt(model, [x])
    name2 = os.path.splitext(weights_name)[0] + '_docker_trt.pth'
    torch.save(model_trt.state_dict(), os.path.join(args.pretrained, name2))


if __name__ == '__main__':
    main()

```

然后测试：

```python
	weights_name = 'hrnet_w48_epoch231_tr_fwiou_0.99038_tr_OA_0.9951_docker_trt.pth'
    model_state_file = os.path.join(args.pretrained, weights_name)
    pretrained_dict = torch.load(model_state_file)
    
    if if_ac:
        model = TRTModule()
        model.load_state_dict(pretrained_dict)
    
    else:
        # build model
        model = eval(config.MODEL.NAME + '.get_seg_model')(config)
        model_dict = model.state_dict()
        # torch.save(pretrained_dict, os.path.join(args.pretrained, "HRNetW48_epoch_229_fwiou_0.97666_OA_0.9880_docker.pth"),
        #            _use_new_zipfile_serialization=False)
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
```

很多时候出现的问题基本都是版本对应不上。多试试几个版本即可。

模型转成功有时候不能用，是因为有一些warning：

```shell
Warning: Encountered known unsupported method torch.nn.functional.interpolate
Warning: Encountered known unsupported method torch.nn.functional.upsample
Warning: Encountered known unsupported method torch.nn.functional.interpolate
Warning: Encountered known unsupported method torch.nn.functional.upsample
Warning: Encountered known unsupported method torch.nn.functional.interpolate
Warning: Encountered known unsupported method torch.nn.functional.upsample
```

看了官方文档：

```python
@tensorrt_converter('torch.nn.functional.interpolate', enabled=trt_version() >= '7.1')
@tensorrt_converter('torch.nn.functional.upsample', enabled=trt_version() >= '7.1')
```

意思是该算子能用的前提是tensorRT版本要高于或者等于7.1





## 提交容器

为了比较方便，把前面的很多操作写了个**shell脚本**（要在`~/.bashrc`的` <<< conda initialize <<<`后面加一行`conda activate torch`。这样就会自动启用torch虚拟环境）。

这样在进行`docker commit`的时候用的CMD就可以比较简单。

有用anaconda：

```shell
source ~/.bashrc
cd /
cd TensorRT-7.1.3.4/python
pip install tensorrt-7.1.3.4-cp36-none-linux_x86_64.whl
cd /torch2trt
python setup.py install
cd /workspace
python run.py /input_path /output_path
```

没用anaconda：

```shell
cd /
cd TensorRT-7.1.3.4/python
pip install tensorrt-7.1.3.4-cp36-none-linux_x86_64.whl
source /etc/profile
source /root/.bashrc
cd /torch2trt
python setup.py install
cd /workspace
python run.py /input_path /output_path
```

反正自己根据需求来啦！



### 提交容器到镜像

采用以下命令：

```shell
docker commit --change="WORKDIR /workspace" -c 'CMD ["bash","run.sh"]' 容器名字 镜像名字:版本号
docker commit --change="WORKDIR /workspace" -c 'CMD ["bash","run.sh"]' znr_hb_yyds znr_hb_1604_102:v1
```

好像有说shell路径什么的问题，所以最好用下面的方法（加了个`./`）：

```shell
docker commit --change="WORKDIR /workspace" -c 'CMD ["bash","./run.sh"]' 容器名字 镜像名字:版本号
docker commit --change="WORKDIR /workspace" -c 'CMD ["bash","./run.sh"]' znr_hb_yyds znr_hb_1604_102:v1
```

这边采用的是覆盖的方法。覆盖之前：

```shell
REPOSITORY                                          TAG                           IMAGE ID       CREATED         SIZE
znr_hb_1604_102                                     v1                            16ff4f40d217   4 hours ago     2.76GB
```

覆盖之后：

```shell
REPOSITORY                                          TAG                           IMAGE ID       CREATED              SIZE
znr_hb_1604_102                                     v1                            a526de01ea60   About a minute ago   15GB
```

可以看到，镜像ID不一样了。然后由于比赛及平台需求，需要对镜像打tag：

```shell
docker tag 镜像号 registry.cn-hangzhou.aliyuncs.com/damonzheng46/znr_hb:版本号
docker tag a526de01ea60 registry.cn-hangzhou.aliyuncs.com/damonzheng46/znr_hb:vtrt_r_1604_102
```

打完tag之后：

```shell
REPOSITORY                                              TAG                               IMAGE ID       CREATED         SIZE
znr_hb_1604_102                                         v1                                a526de01ea60   4 minutes ago   15GB
registry.cn-hangzhou.aliyuncs.com/damonzheng46/znr_hb   vtrt_r_1604_102                   a526de01ea60   4 minutes ago   15GB
```

可以看到镜像ID是一样的，这其实是同一个。

### 提交到官网

先提交到阿里云，首先登录一下：

```shell
docker login --username=用户名 registry.cn-hangzhou.aliyuncs.com
docker login --username=damonx registry.cn-hangzhou.aliyuncs.com
```

登录成功有以下提示：

```shell
WARNING! Your password will be stored unencrypted in /root/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
```

然后push：

```shell
docker push 镜像名字:版本号
docker push registry.cn-hangzhou.aliyuncs.com/damonzheng46/znr_hb:vtrt_r_1604_102
```

结果：

```shell
[root@dbcloud-All-Series  /home/dbcloud]$docker push registry.cn-hangzhou.aliyuncs.com/damonzheng46/znr_hb:vtrt_r_1604_102
The push refers to repository [registry.cn-hangzhou.aliyuncs.com/damonzheng46/znr_hb]
d01fee0276e2: Pushed 
bdeefed578e8: Pushed 
1d3b056d45fd: Pushed 
7c3df08f66e5: Pushed 
850a057a2cb5: Pushed 
6621e9acaaa7: Pushed 
0e72e68025bd: Pushed 
62095c830902: Pushed 
0b4d7cfd9110: Pushed 
3f9222e218bf: Pushed 
1251204ef8fc: Pushed 
47ef83afae74: Pushed 
df54c846128d: Pushed 
be96a3f634de: Pushed 
vtrt_r_1604_102: digest: sha256:be35ae23737f7a9b32e55b236f77160cdd5cce5298ffc229d9822b223e7d4400 size: 3262
```

然后就可以提交了。

## 注意事项

1. 乱拳打死老师傅，反正大概是这个意思这个流程。

2. 如果更改了环境变量，虽然会执行`source ~/.bashrc`，但是是在启动命令CMD之后才执行。所以如果CMD运行需要用到这个环境，那么就会报错。（[参考链接](https://blog.csdn.net/u010483897/article/details/95363587)）

3. 解决2的方法是直接写个shell脚本，让容器一开始就直接运行.sh文件。直接把环境什么的都弄一遍就完事了。

4. 安装opencv出错，可以用`pip install --upgrade pip`更新一下。

5. 有时候anaconda用不了，可以不用anaconda。但是不用anaconda的时候，直接在外面安装torchvision的时候会报错，[参考链接](https://zhuanlan.zhihu.com/p/503678336)。用下面两条命令：

   ```shell
   apt-get install -y libjpeg-dev zlib1g-dev
   pip install -i https://mirrors.aliyun.com/pypi/simple/ Pillow
   ```

   

6. 出错`ModuleNotFoundError: No module named '_lzma'`，[参考链接](https://zhuanlan.zhihu.com/p/404162713)。

   ```shell
   apt-get install liblzma-dev -y
   pip install backports.lzma
   vim /usr/local/python3.6/lib/python3.6/lzma.py
   ```

   ```shell
   #修改前
   from _lzma import *
   from _lzma import _encode_filter_properties, _decode_filter_properties
   
   #修改后 
   try:
       from _lzma import *
       from _lzma import _encode_filter_properties, _decode_filter_properties
   except ImportError:
       from backports.lzma import *
       from backports.lzma import _encode_filter_properties, _decode_filter_properties
   ```

   

7. `source ~/.bashrc`最好都写成`source /root/.bashrc`，因为docker里面可能有分什么这个用户跟所有用户，不是很懂。

## 提交之后出现的问题

### 问题1

提交之后出现最顽固的问题就是：

```shell
from .tensorrt import * ImportError: libnvinfer.so.7: cannot open shared object file: No such file or directory
```

这个其实本地配置的时候添加了环境变量就可以解决，但是别人拉取镜像之后好像不行。

尝试过用run.sh的时候添加变量或者激活`source~/.bashrc`，但还是没有用。查了一下发现，有人[回答](https://qastack.cn/programming/28722548/updating-path-environment-variable-permanently-in-docker-container)：

> 因此，设置.bashrc文件似乎仅在作为交互式终端运行时才起作用，这很有意义，因为它将运行用户的默认外壳程序。除非您通过bash发送命令，否则使用该文件将无法在容器内运行命令。

说通过bash发送指令才可以用，然是我整个run.sh就是bash运行的，不对，是source运行的，应该没问题才对。但是我提交过anaconda环境的，也是本地配置好.bashrc激活torch虚拟环境，提交之后还是不行。所以估计就是.bashrc没办法再一次source。

尝试方法：

1. `vim /etc/profile`添加环境变量并激活`source /etc/profile`。（[据说](https://www.jianshu.com/p/085b742adf6e)这样可以永久修改，个人认为这种比较靠谱，哎，学艺不精，笨死算了。）
2. 直接在run.sh中添加`export LD_LIBRARY_PATH=/TensorRT-7.1.3.4/lib:$LD_LIBRARY_PATH`。（保证运行的时候能修改）
3. 在run.sh中使用`source /root/.bashrc`。（保证运行的时候再激活一下）

### 问题2

其实问题1所有办法都是可以解决的。之所以提交之后用不了，是因为官网默认运行指令就是`python run.py /input_path /output_path`，而不管的启动命令。所以其实`run.sh`是没有被运行的。因此环境变量没有被激活。也就是说，容器提交成镜像只能用：

```shell
docker commit --change="WORKDIR /workspace" -c 'CMD ["python","run.py","/input_path","/output_path"]' 容器名字 镜像名字:版本号
docker commit --change="WORKDIR /workspace" -c 'CMD ["python","run.py","/input_path","/output_path"]' znr_hb_best znr_1604_102:v1
docker commit --change="WORKDIR /workspace" -c 'CMD ["python","run.py","/input_path","/output_path"]' sxz_hb sxz_hb:v1
```

最后可以采用`docker run -it --gpus all --name try znr_1604_102:v1`试看看能不能跑，能的话官网应该就没问题了。一定要加`--gpus all`，因为tensorrrt是要用到cuda的，不然还是会报错。



## 测试

### 测试结果

在docker内用6张图像测试w48（512，1024，2048各两张）

1. 不加速：时间`4.883496522903442`，background`0.99759213`，seaice`0.9263046`，fwiou`0.9953985145972329`，acc`0.9976628621419271`。
2. 加速：时间`11.41271162033081`，background`0.99759213`，seaice`0.9263046`，fwiou`0.9953985145972329`，acc`0.9976628621419271`。

---

在3090服务器用1500张图像测试w48：

1. 不加速：时间`132.32292246818542`，background`0.99572575`，seaice`0.96188904`，fwiou`0.9923679904416806`，acc`0.99614194723276`。
2. 加速：时间`58.74643111228943`，background`0.99572571`，seaice`0.96188851`，fwiou`0.992367905741876`，acc`0.9961419133745002`。

在3090服务器用1500张图像测试w48——bs8：

1. 不加速：时间`49.82384920120239`，background`0.99572577`，seaice`0.96188925`，fwiou`0.9923680333170153`，acc`0.9961419698049331`。
2. 加速：时间`53.353134632110596`，background`0.99572354`，seaice`0.96186903`，fwiou`0.992364016478023`，acc`0.9961399439523911`。
3. MS+resize：时间`265.7972893714905`，background`0.99667527`，seaice`0.97027759`，fwiou`0.9940557211234502`，acc`0.9970007902066383`。

在3090服务器用1500张图像测试w48——bs16：

1. 不加速：时间`52.860244274139404`，background`0.99572589`，seaice`0.96189027`，fwiou`0.9923682393156465`，acc`0.9961420751417412`。
2. 加速：模型太大。

---

在3090服务器用1500张图像测试w30：

1. 不加速：时间`42.84110903739929`，background`0.9938186`，seaice`0.946152`，fwiou`0.9890884508070953`，acc`0.9944242029735557`。
2. MS+resize：时间`426.5595397949219`，background`0.99562661`，seaice`0.96149447`，fwiou`0.9922395400119673`，acc`0.9960571868414945`。

---

在3090服务器用1500张图像测试w18：

1. 不加速：时间`125.87282514572144`，background`0.97878368`，seaice`0.83273363`，fwiou`0.9642905342269888`，acc`0.9808105698233761`。
2. 加速：时间`45.183743953704834`，background`0.9788143`，seaice`0.83292892`，fwiou`0.9643374896461762`，acc`0.98083818311522`。
3. MS+resize：时间`351.6038863658905`，background`0.98311778`，seaice`0.86283783`，fwiou`0.9711819066210647`，acc`0.984738547421066`。

在3090服务器用1500张图像测试w18——bs8：

1. 不加速：时间`43.107840061187744`，background`0.97878362`，seaice`0.83273368`，fwiou`0.964290478025916`，acc`0.9808105152739576`。
2. 加速：时间`33.90340065956116`，background`0.97881384`，seaice`0.83292597`，fwiou`0.964336787606051`，acc`0.980837773054074`。
3. MS+resize：时间`121.04310512542725`，background`0.98311639`，seaice`0.8628283`，fwiou`0.9711797078607061`，acc`0.9847372927844407`。

在3090服务器用1500张图像测试w18——bs16：

1. 不加速：时间`52.860244274139404`，background`0.99572589`，seaice`0.96189027`，fwiou`0.9923682393156465`，acc`0.9961420751417412`。
2. 加速：没测。

### 测试结论

1. batch_size一定要有的，为1的时候很慢，至于选多少不一定，理论上越大越快，但是实验中好像还取决于模型的大小。





### 测试结果2

在3090服务器用1500张图像测试w30：

1. 有强制resize-0000-bs8：时间`42.84110903739929`，background`0.9938186`，seaice`0.946152`，fwiou`0.9890884508070953`，acc`0.9944242029735557`。线上`97.3555`，时间`100`。
2. 无强制resize-1010-bs1：时间`426.5595397949219`，background`0.99562661`，seaice`0.96149447`，fwiou`0.9922395400119673`，acc`0.9960571868414945`。线上`98.4431`，时间`274`。
3. 无强制resize-0000：
4. 无强制resize-1010加速-bs1：时间`266.207102060318`，background`0.900769504`，seaice`0`，fwiou`0.8113864583896148`。
5. 有强制resize-0000加速-bs8：时间`41.23596382141113`，background`0.99382127`，seaice`0.9461745`，fwiou`0.9890930816616152`，acc`0.9944266106720272`。（docker报错）
6. 有强制resize-0000加速-bs1：时间`49.649797201156616`，background`0.99381804`，seaice`0.94614769`，fwiou`0.989087514409787`，acc`0.9944236969806739`。（线上报错，output路径错误）
7. 有强制resize-1010-减scares-bs16：（没交）
8. 有强制resize-0000-bs16：

