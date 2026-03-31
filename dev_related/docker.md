
- [安装docker](#安装docker)
- [管理docker服务](#管理docker服务)
- [容器相关命令](#容器相关命令)
- [镜像相关命令](#镜像相关命令)
- [volume相关](#volume相关)
- [network相关](#network相关)
- [Dockerfile](#dockerfile)
  - [Dockerfile 核心指令](#dockerfile-核心指令)
  - [Dockerfile 构建](#dockerfile-构建)

# 安装docker

**普通安装docker**

```shell
# 安装依赖
apt update
apt install -y apt-transport-https ca-certificates curl software-properties-common
# 添加官方密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
# 添加docker仓库
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
# 再次更新apt软件包索引
apt update
# 安装 Docker Engine
apt install -y docker-ce docker-ce-cli containerd.io
# 验证docker是否安装成功
docker version
```

**支持 nvidia**

```shell
# 要使用`--gpus all`需要额外安装`NVIDIA Container Toolkit`
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
```

**安装 docker compose**

```shell
# 安装docker compose（若需要）
mkdir -p ~/.docker/cli-plugins/
curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
docker compose version
```

# 管理docker服务

**管理docker服务**

```shell
systemctl [command] docker  # 新版
service docker [command]    # 旧版
# command 如下：
    # start: 启动
    # restart: 重启
    # stop: 停止
    # status: 查看状态
```

**开机自启**

```shell
systemctl enable docker     # 设置docker开机自启
systemctl disenable docker  # 取消docker开机自启
```

**查看docker磁盘使用情况**

```shell
docker system df [-v]
```

**清理build缓存**

```shell
docker builder prune
```

**查看容器/镜像的详细信息**

```shell
docker inspect [container_name|image_id]
```


# 容器相关命令

**运行容器**

```bash
docker run --gpus all -it -d --rm \
    -p <host_port:container_port> \
    -v <host_path:container_path> \
    -e <var_name=val_value> \
    --platform=<platform> \
    --net=host \
    --network <network_name> \
    --ip <custom_ip> \
    --shm-size=10gb \
    --name <container_name> <image_name:image_tag> [/bin/bash]

# --gpus all：允许容器使用所有gpu
# -it：交互模式
# -d：后台运行
# --rm：运行结束删除
# -p：端口映射（有-p就不能有--net=host）
# -v：数据卷映射
# -e：环境变量
# --platform=<platform>：指定容器架构（linux/amd64, linux/arm64）
# --net=host：容器直接使用宿主机的网络。（与-p、--network等字段是冲突）
# --network：采用哪个docker网络，允许同一个网络间的容器互相通信
# --ip：固定ip（必须跟--network配合使用）
# --shm-size=10gb：共享内存大小设置为10GB。适用于TensorFlow、PyTorch等需要大共享内存的深度学习任务，否则默认/dev/shm只有64MB，可能导致OOM（内存不足）。
# --name：容器名称
```

**查看容器**

```shell
# 1. 查看容器占用资源
docker stats [--no-stream] [container_name|container_id]    # --no-stream：一次性快照，不实时刷新

# 2. 查看容器信息
docker ps [OPTIONS]

# -s：查看容器占用磁盘（其中size是本容器在它的镜像基础上写入了多少新内容，virtual是包括了基础镜像大小）
# -a：查看所有容器
# -q：只显示容器id
# -f：过滤条件：`docker ps -f “name=<container_name>”`
# -n：最近创建的n个：`docker ps -n 3`
# --format "{{.字段名}}"`：格式化显示，如`docker ps --format "{{.ID}},{{.Image}},{{.Names}}"`
```

**管理容器**

```shell
docker [command] [container_name|container_id]
# command如下：
    # stop: 停止容器
    # restart: 重启容器
    # rm [-f]: 删除容器
    # pause: 暂停容器（短暂释放资源时使用）
    # unpause: 取消暂停
```

**进入容器**

```shell
# 1. attach 进入（不推荐）
docker attach <container_name>

# 2. exec 进入（推荐）
docker exec -it <container_name> bash

# attach 将连接到容器的主进程，可能会影响容器的运行（如使用 Ctrl+C 可能会停止容器）
# exec 启动一个新的进程，而不是连接到已有的主进程，不会影响容器的主进程，退出新进程不会停止容器
# 也就是说，假如这个容器有启动命令，一直在前台运行某个服务，attach进去之后，其实没办法操作，只能Ctrl+C停止进程，而容器一般都使用-d -rm 之类的命令启动的，这么做就会使容器直接停止并删除。而使用exec进去之后是新开了一个进程，并不会影响主进程的运行，因此很适合进入有启动命令的容器查看相关信息。（总之推荐用exec！！！）
```

**查看正在运行的docker的日志（若有）**

```shell
docker logs <container_name> --tail 10 -f
    # --tail 10: 查看最后10行
    # -f: 实时刷新
```

**容器与宿主机文件传输**

```shell
# 宿主机传输到容器
docker cp <本机文件路径> <容器名字/id>:<容器内路径>
# 容器传输到宿主机
docker cp <容器名字/id>:<容器内路径> <本机文件路径>
```

**容器提交到镜像**

```shell
docker commit <container_id/container_name> <newimage_name:tag> \
    -a "damonzheng" \
    -m "update base image with new configurations" \
    --change <change_content>

# -a, --author：指定作者
# -m, --message：添加注释
# --change：修改Docker配置（可以多行）
    # 修改CMD：--change 'CMD ["bash", "start.sh"]'
    # 修改ENTRYPOINT：--change 'ENTRYPOINT ["python", "app.py"]'
    # 修改环境变量ENV：--change "ENV APP_ENV=production"
    # 修改工作目录WORKDIR：--change "WORKDIR /data/code"
    # 修改暴露端口EXPOSE：--change "EXPOSE 8080"
```

# 镜像相关命令

**查看镜像**

```shell
# 1. 查看镜像信息
docker images [OPTIONS]
    # -a: 显示所有镜像（包括中间层）
    # -q: 只显示镜像ID
    # --digests: 显示镜像的摘要信息

# 2. 查看镜像构建过程
docker history <image_id>
```

**管理镜像**

```shell
docker pull <image_name:tag>    # 拉取
docker push <image_name:tag>    # 推送
docker rmi <image_name:tag>     # 删除
    # -f: 强制删除
    # docker rmi $(docker images -q)`: 删除所有镜像
```

**镜像打标签**

```shell
docker tag <image_id> <target_image:tag>
```

**从Dockerfile构建镜像**

```shell
docker build -t <image_name:image_tag> [-f </path/to/Dockerfile>] .
    # -t: 指定tag，默认为latest
    # -f: 指定Dockerfile文件，没指定默认为当前路径
    # .：决定了docker引擎在构建镜像时可以使用的文件和目录
```

**保存与加载镜像**

```shell
docker save -o <file_path/file_name.tar> <image_name:tag>   # 保存镜像到本地文件
docker load -i <image_name.tar>     # 加载本地保存的镜像
```


# volume相关

**创建volume**

```shell
docker volume create <volume_name>
```

**查看volume信息**

```shell
# 查看所有volume
docker volume ls
# 查看某个volume详细信息
docker volume inspect <volume_name>
```

**删除volume**

```shell
docker volume rm <volume_name>  # （正在使用则无法删除）
```

**volume 和 bind mount**：

| 特性             | volume                                                       | bind mount（挂载宿主机目录）       |
| ---------------- | ------------------------------------------------------------ | ---------------------------------- |
| **创建方式**     | `docker volume create` 或 `-v <colume_name>:/<container_path>` | `-v <host_path>:/<container_path>` |
| **数据存储位置** | Docker 自动管理，比如 `/var/lib/docker/volumes/...`          | 指定的宿主机路径                   |
| **安全性**       | 更安全，容器不会直接接触宿主机目录结构                       | 容器可能访问敏感宿主机路径         |
| **性能**         | 一般比 bind mount 更快（尤其是 Linux）                       | 性能略低，受文件系统影响           |
| **便携性**       | 高，可在不同主机间迁移（配合 `volume inspect` + tar）        | 低，依赖宿主机目录结构             |
| **典型用途**     | 数据持久化、数据库卷、配置存储                               | 本地开发时同步代码、调试数据       |

# network相关

**创建network**

```shell
docker network create <network_name>
```

**查看network信息**

```shell
# 查看所有network
docker network ls
# 查看某个network详细信息（加上` | grep Name`可以查看当前网络有哪些容器）
docker network inspect <network_name>
```

**容器与network的连接跟断开**

```shell
docker network connect <network_name> <container_name_or_id>        # 连接
docker network disconnect <network_name> <container_name_or_id>     # 断开
```

**删除network**

```shell
docker network rm <network_name>
```

**使用docker network的好处**

1. 容器自动内网互通：同一个 network 下的容器，自动能用 容器名当域名互相访问，不用记 IP 地址
2. 自定义IP段：自定义网络时可以指定 IP 段、子网掩码，方便跟现有基础设施融合
3. 隔离性更强：不同的 network 之间 默认互相隔离，安全性高，防止容器乱串
4. 支持多网卡容器：一个容器可以连到多个 network，相当于多块网卡，复杂应用（如代理服务器）必备
5. 跨主机扩展：使用 Swarm / Kubernetes，可以把 network 跨宿主机打通，做跨机集群
6. 内建负载均衡：多个容器用同一个名字（服务名）注册时，docker network 会自动轮询负载均衡请求（bridge overlay支持）

# Dockerfile

**简单理解**

它本质是一个声明式的构建流程，描述：从哪个基础镜像 -> 做哪些操作 -> 最终启动什么程序

**执行本质**

Dockerfile 是一层一层构建的（layer）：

```shell
FROM ubuntu
RUN apt update
RUN apt install -y python3
COPY . /app
```

会变成：

```shell
Layer1: ubuntu
Layer2: apt update
Layer3: install python3
Layer4: copy files
```

核心特点：
- 每行一个 layer
- 有缓存（cache）
- **改一行会导致后面的 layer 全部失效**

## Dockerfile 核心指令

**FROM（必须有）**

指定基础镜像（必须是第一行），如：

```shell
FROM ubuntu:20.04
```

**SHELL**

改变 Dockerfile 中 RUN 使用的默认 shell，默认是 `/bin/sh -c`，可以用：

```shell
SHELL ["/bin/bash", "-lc"]
# 后面的 RUN 都用：/bin/bash -lc "<command>"
    # -l: 让 bash 变成“登录 shell”，会加载：/etc/profile、~/.bash_profile、~/.bashrc（间接）
    # -c: 表示执行后面的一段命令字符串
```

**RUN（执行命令）**

在构建镜像时执行命令，每个 RUN 都会生成一层，**推荐合并写（减少层数）**：

```shell
# 推荐：
RUN apt update && apt install -y curl vim

# 不推荐：
RUN apt update
RUN apt install -y curl
```

**COPY（复制文件）**

把宿主机文件复制到镜像，支持目录，不会解压，如：

```shell
# 把当前路径下所有文件 cp 到镜像的 /app 中
COPY . /app
```

**ADD（增强版 COPY）**

比 COPY 多两个能力：
- 自动解压
- 支持 URL

但是，优先用 COPY，除非需要解压，如：

```shell
ADD file.tar.gz /app
```

**WORKDIR（工作目录）**

设置后续命令的执行目录，相当于 `cd /app`，如：

```shell
WORKDIR /app
```

**CMD（容器启动命令）**

容器启动时默认执行，特点：
- 只能有一个（最后一个生效）
- 可以被 `docker run` 覆盖

```shell
CMD ["python", "app.py"]
```

**ENTRYPOINT（更强的启动控制）**

特点：
- 不容易覆盖

```shell
ENTRYPOINT ["python"]
CMD ["app.py"]
# 效果：python app.py
```

**ENV（环境变量）**

设置环境变量，如：

```shell
ENV APP_ENV=production
```

**EXPOSE（端口声明）**

声明容器端口（仅文档作用），**不会真正开放端口！**，如：

```shell
EXPOSE 8080
```

**VOLUME（挂载点）**

定义匿名卷，如：

```shell
VOLUME /data
```

**ARG（构建参数）**

构建时，`docker build --build-arg VERSION=1.0 .`：

```shell
ARG VERSION=1.0
```

**USER（切换用户）**

提高安全性（生产必须），如：

```shell
USER nobody
```

**LABEL（元信息）**

```shell
LABEL author="you"
```

## Dockerfile 构建

如果是比较复杂的镜像构建，需要去容器中不断尝试的。可以先到容器中配置好，然后选择以下方式构建：
1. 把 shell 历史命令导出：`history` 或者 `cat ~/.bash_history`，根据历史命令写 Dockerfile
2. 用 docker diff 反推修改：用 `docker diff <container>` 查看改动，这个方法可以看到“手工改了哪些文件”

结合上面两个就可以构建一个比较复杂的镜像了。下面给一个很简单的 Dockerfile 模板：

```shell
FROM ubuntu:20.04

# 1. 基础工具
RUN apt update && apt install -y \
    curl \
    vim \
    git

# 2. 语言环境
RUN apt install -y python3 python3-pip

# 3. 依赖
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 4. 代码
WORKDIR /app
COPY . .

# 5. 启动
CMD ["python3", "main.py"]
```

稍微复杂一点的可以看看我的另一个 [Dockerfile](https://github.com/NairongZheng/openclaw_gen_data/blob/main/Dockerfile)
