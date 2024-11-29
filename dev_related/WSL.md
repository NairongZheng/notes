[toc]

# 参考链接

[玩转 Windows 自带的 Linux 子系统](https://zhuanlan.zhihu.com/p/258563812)

[win10利用WSL2安装docker的2种方式](https://zhuanlan.zhihu.com/p/148511634)

# WSL安装配置

## 本地启动WSL功能

1. `win+s`，搜索 PowerShell，右键管理员身份运行
2. 输入命令，启用`适用于Linux的Windows子系统`功能：`dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`

## 下载并安装WSL

1. 选择版本：[旧版 WSL 的手动安装步骤 | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/wsl/install-manual) 。或者可以直接在微软商店下载。不过微软商店默认安装在C盘。（[自定义WSL的安装位置](https://zhuanlan.zhihu.com/p/263089007)）
2. 下载完是`.appx`后缀的文件，将其后缀改成`.zip`并放到要安装的路径，然后解压。（[Windows10/11 三步安装wsl2 Ubuntu20.04（任意盘）](https://zhuanlan.zhihu.com/p/466001838)）（可以从[网盘](https://pan.baidu.com/s/1WCW1vzx2ZqOzwZ7hnsXKJw?pwd=wwaq)下载：）
3. 解压完后运行exe文件，即可将WSL安装到当前路径。
4. 需要注意的是安装目录的磁盘不能开`压缩内容以便节省磁盘空间`选项，否则会报错。可以右键`文件夹-->属性-->常规-->高级`找到并关闭这个选项。
5. 以上几个步骤可以用以下命令来进行（下载好像会比较快）：
   1. 下载（下面给的是20.04）
   2. 改后缀
   3. 解压
   4. 安装

```bash
Invoke-WebRequest -Uri https://wsldownload.azureedge.net/Ubuntu_2004.2020.424.0_x64.appx -OutFile Ubuntu20.04.appx -UseBasicParsing
Rename-Item .\Ubuntu20.04.appx Ubuntu.zip
Expand-Archive .\Ubuntu.zip -Verbose
cd .\Ubuntu\
.\ubuntu2004.exe
```

## WSL设置

### 多个版本选择

1. 查看当前有多少个wsl：`wsl -l -v`
2. 进入某个wsl：`wsl -d 版本`
3. 查看ubuntu版本：`cat /proc/version`

### WSL设置账号密码

1. 安装完在命令行直接设置
2. 有时候错过了，就重新设置一下（[Windows下WSL的root密码忘记解决办法](https://blog.csdn.net/xdg15294969271/article/details/122374964)）
3. 每次进入root都需要输入密码，可以设置一下某个用户进入不需要密码：`sudo echo "<user_name> ALL=(ALL:ALL) NOPASSWD: ALL" >>/etc/sudoers`

### WSL路径

1. [windows10 Linux子系统(wsl)文件目录](https://blog.csdn.net/x356982611/article/details/80077085)
2. WSL 1的路径：`C:\Users\用户名\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs`
3. WSL 2的路径：被映射到`\\wsl$`。在文件夹上面的搜索栏输入`\\wsl$`即可

# Docker与Anaconda

## Docker安装

运行以下命令：

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo service docker start
```

检查docker安装正常：

```bash
service docker status
ps aux|grep docker
```

## 容器环境配置与制作镜像

### 正常配置

**（1）基础镜像拉取与容器运行**

1. 拉一个镜像（这边从官网拉）：`docker pull ubuntu`（会自动拉取最新的ubuntu，当然也可以指定版本号）
2. 运行容器：`docker run -it --name <container_name> <image_name:image_tag> bash`
3. 安装一些必要的工具：

```bash
apt-get update
apt-get install sudo
apt-get install wget
apt-get install vim
apt-get install ssh
apt-get install screen
apt-get install git
```

**（2）Anaconda安装**

anaconda官网下载，链接：[Free Download | Anaconda](https://www.anaconda.com/download#downloads)

直接复制链接，到Linux中下载即可。注意要把目录切换到`/opt`中（[Linux 软件安装到 /usr，/usr/local/ 还是 /opt 目录？](https://blog.csdn.net/CPOHUI/article/details/103215252)）：

```bash
cd /opt
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```

下载太慢的话可以用清华源：[清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)

这里是要在容器中配置环境，所以将下载好的anaconda拷贝到docker容器的opt里面再安装（如果是在本地WSL里配环境，那直接在/opt装就行）。拷贝命令：

```bash
docker cp /opt/Anaconda3-2023.09-0-Linux-x86_64.sh <container_name>:/opt
```

开始安装。

```bash
sh Anaconda3-2023.09-0-Linux-x86_64.sh
```

注意：

1. 一直按enter，其中有一个是否接收license记得输入`yes`
2. 安装路径好像不是默认在放`Anaconda3-2023.09-0-Linux-x86_64.sh`的地方，所以其中还有一步要选择安装路径的，可以用：`/opt/anaconda3`
3. 是否初始化也要选`yes`

如果改变了默认安装路径的话，初始化好像是不会成功的（可以使用`/opt/anaconda3/bin/conda init`进行初始化），或者自己添加环境变量（[Anaconda 多用户共享安装（Ubuntu）](https://zhuanlan.zhihu.com/p/570747928#:~:text=%E8%BF%90%E8%A1%8C%E4%B8%8B%E8%BD%BD%E7%9A%84%E6%96%87%E4%BB%B6%E3%80%82%20%E4%BD%A0%E5%8F%AF%E8%83%BD%E6%83%B3%E4%B8%BA%E5%AE%83%E5%A2%9E%E5%8A%A0%E6%89%A7%E8%A1%8C%E6%9D%83%E9%99%90%EF%BC%8C%E9%80%9A%E8%BF%87%20chmod%20%2Bx%20%E3%80%82%20%E5%9B%9E%E8%BD%A6%E5%BC%80%E5%A7%8B%E5%AE%89%E8%A3%85%20%E9%98%85%E8%AF%BB%E5%B9%B6%E8%BE%93%E5%85%A5%20yes,%2Fopt%2Fanaconda3%20%E3%80%82%20%E4%B8%8D%E8%A6%81%E6%94%BE%E5%9C%A8%20root%20%E6%A0%B9%E7%9B%AE%E5%BD%95%EF%BC%88%20~%20%EF%BC%89%E4%B8%8B%20%E3%80%82) ）：

1. 编辑文件：`vim ~/.bashrc`
2. 在文件末尾添加：`export PATH=/opt/anaconda3/bin:$PATH`
3. 命令行更新一下：`source ~/.bashrc`

**（3）虚拟环境配置**

假设要配置一个名为`dev`的环境，并且在里面安装`pytorch`

1. 创建虚拟环境：`conda create -n dev python=3.8`
2. 激活环境（最新版的conda好像没有activate，所以只能用source）：`source activate dev`
3. 安装cpu版本的torch（用清华源）：`pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple`（如果要下载gpu版本的需要用别的命令）

**（4）提交容器到镜像**

容器提交：`docker commit <container_name> <image_name:image_tag>`。可以看看`docker commit`和`docker build`的区别：[docker commit 和docker build](https://blog.csdn.net/alwaysbefine/article/details/111375658)

### 使用Dockerfile配置

上面的流程可以用以下Dockerfile配置（被注释掉的是使用miniconda）：`docker build -t <image_name:image_tag> .`

一些包提前下载：

1. anaconda：官网或者[清华源](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)
2. miniconda：官网或者[清华源](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)
3. gdal：[anaconda官方库](https://anaconda.org/conda-forge/gdal/files?page=2&version=2.4.4)

（下面的Dockerfile中更换的apt源好像用不了了，可以另外找或者不更换！）

```bash
FROM ubuntu:20.04
LABEL maintainer="damonzheng46@gmail.com" date="20231221"
SHELL ["/bin/bash","-c"]
COPY linux-64_gdal-2.4.4-py38hfe926b7_1.tar.bz2 /tmp
COPY Anaconda3-2023.09-0-Linux-x86_64.sh /tmp
# COPY Miniconda3-py310_23.5.1-0-Linux-x86_64.sh /tmp
RUN \
    # 更换apt源
    cp /etc/apt/sources.list /etc/apt/sources.list.bak \
    && sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list \
    # 安装一些工具并配置ssh
    && apt-get update \
    && apt-get -y install sudo \
    && apt-get -y install wget \
    && apt-get -y install vim \
    && apt-get -y install screen \
    && apt-get -y install git \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install ssh \
    && apt-get -y install openssh-server \
    && apt-get -y install openssh-client \
    # && sed -i 's/UsePAM yes/UsePAM no/g' /etc/ssh/sshd_config \ 
    # && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config \
    && sed -i '$a\UsePAM no' /etc/ssh/sshd_config \
    && sed -i '$a\PermitRootLogin yes' /etc/ssh/sshd_config 
RUN \
    # 下载安装anaconda并配置虚拟环境
    cd /tmp \
    && sh Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/anaconda3 \
    && /opt/anaconda3/bin/conda init \
    && export PATH="/opt/anaconda3/bin":$PATH \
    && conda create -n dev python=3.8 -y \
    && source activate dev \
    && pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install tqdm yacs opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && conda install linux-64_gdal-2.4.4-py38hfe926b7_1.tar.bz2 \
    && conda install gdal \
    # 清理
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*
# RUN \
#     # 下载安装miniconda并配置虚拟环境
#     cd /tmp \
#     && sh Miniconda3-py310_23.5.1-0-Linux-x86_64.sh -b -p /opt/miniconda3 \
#     && /opt/miniconda3/bin/conda init \
#     && export PATH="/opt/miniconda3/bin":$PATH \
#     && conda create -n dev python=3.8 -y \
#     && source activate dev \
#     && pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple \
#     && pip3 install tqdm yacs opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple \
#     && pip3 install opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple \
#     && conda install linux-64_gdal-2.4.4-py38hfe926b7_1.tar.bz2 \
#     && conda install gdal \
#     # 清理
#     && apt-get clean \
#     && rm -rf /tmp/* /var/tmp/*

EXPOSE 22

CMD ["/bin/bash"]
```

## 上传镜像

可以将镜像传到线上平台，之后换机器就不需要重新配环境，可以直接使用。参考链接：[docker使用与镜像提交](https://blog.csdn.net/a264672/article/details/123119141)

# “远程”连接相关（重要！）

## WSL配置

1. 查看正在用的端口号，该杀杀：`netstat -tanlp`
2. 修改ssh的配置文件（注意端口号，注意看注释）：

```bash
vim /etc/ssh/sshd_config
#填加以下内容：
Port 20 # 为了给下面的Xshell连接WSL开放
Port 4611 # 为了给下面VSCode连接WSL上的容器开放
PermitRootLogin yes #允许root用户使用ssh登录

LoginGraceTime 120
StrictModes yes
```

1. 重启ssh服务，以下都可以：

```bash
/etc/init.d/ssh restart
sudo service ssh --full-restart
```

1. 添加root密码：`passwd`
2. 做连接测试：`ssh root@127.0.0.1 -p 端口号`
3. 查看ip：`ifconfig`
   1. 会有两个ip，都有用
   2. `172.19.2.87`：远程连接使用，我的是这个
   3. `127.0.0.1`：本地连接使用

## 容器配置

1. 从上面构建的镜像或者拉取的镜像启动容器：`docker run -it -p 4611:4611/tcp -v /mnt:/mnt --name damonzheng damonzheng_image:v1 bash`
   1. `-p`是端口映射：`-p 主机端口:容器端口`（上面WSL开放了4611，下面再开放容器的4611，然后在这边做端口映射，下面再配置VSCode就可以通过访问WSL的4611访问到容器的4611从而进入到容器中）
   2. `-v`是数据卷映射：`-v 主机目录:容器目录`
   3. 因为WSL2数据默认是`/mnt`跟windows映射的，所以docker容器映射到linux的`/mnt`就可以读到windows的磁盘了。这样，数据代码都可以放在本地主机，docker只当环境使用。
2. 修改ssh的配置文件（注意端口号，注意看注释）：

```bash
vim /etc/ssh/sshd_config
#填加以下内容：
Port 4611 # 要跟容器启动时开放端口一样
PermitRootLogin yes #允许root用户使用ssh登录
```

1. 重启ssh服务：`/etc/init.d/ssh restart`
2. 添加root密码：`passwd`
3. 做连接测试：`ssh root@127.0.0.1 -p 端口号`

## Xshell连接WSL

配置如下：

```bash
主机：172.19.2.87 # 上面ifconfig查到的ip
端口号：20 # WSL配置中开放的端口
用户名：root
密码： # WSL配置中设置的passwd密码
```

## VSCode连接WSL

只要在vscode下载插件wsl，然后把远程资源管理器从`远程(隧道/SSH)`改成`WSL 目标`即可。做了数据映射其实没必要连到容器里了，直接连wsl就可以了。

（用ssh配置的办法一直连不上，不知道为什么！谁教教我啊啊啊啊啊！）

## VSCode连接WSL里/远程服务器上的docker容器

使用docker容器里的环境和做完数据映射的本地代码数据做开发，可以直接在本地修改代码，在docker运行。但是这样没办法用vscode逐行调试，只能查看log信息调试。

若要用vscode逐行调试，还是需要直接连接到WSL里面的docker容器，直接用容器里面的环境和映射的代码数据。方法如下：

远程资源管理器用`远程(隧道/SSH)`，然后配置`C:\Users\用户名\.ssh\config`如下：

```bash
Host ubuntu2004
    HostName 127.0.0.1 # 上面ifconfig查到的ip
    User root
    Port 4611 # 容器映射出来的端口
```

有些通过跳板机之类的需要key，大概配置如下：

```bash
Host ******
    HostName 192.168.**.**
    User root
    IdentityFile C:\Users\***\.ssh\vscode.key
    ProxyCommand C:\Windows\System32\OpenSSH\ssh.exe -q -W %h:%p jumpserver
    Port 60011
```

## 一些问题

1. WSL起ssh出现问题`sshd: no hostkeys available -- exiting`：[参考链接](https://blog.csdn.net/p1456230/article/details/120922515)
2. 连接测试出现问题`Permission denied (publickey)`：[参考链接](https://blog.csdn.net/yjk13703623757/article/details/114936739)
3. “设置SSH主机：XXX：正在本地下载VS Code服务器”时间过长：[链接1](https://www.cnblogs.com/ziangshen/articles/17741402.html)，[链接2](https://blog.csdn.net/xihuanyuye/article/details/124901961)（同时参考！）
4. VSCode不断要求输入密码：[参考链接](https://blog.csdn.net/Mr_Cat123/article/details/107432070)
5. 连接测试出问题：假设`ssh root@127.0.0.1 -p 20`出问题`It is also possible that a host key has just been changed..…..`那么可以查看文件`/root/.ssh/known_hosts`，更新里面相应的key，命令如`ssh-keygen -R [127.0.0.1]:20`