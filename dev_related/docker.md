
- [其他命令](#其他命令)
- [容器相关命令](#容器相关命令)
- [镜像相关命令](#镜像相关命令)


# 其他命令
1. 启动docker服务：`service docker start`
2. 重启docker服务：`systemctl restart docker`
3. 查看docker磁盘使用情况：`docker system df`
4. 清理build缓存：`docker builder prune`
5. 查看容器/镜像的详细信息：`docker inspect [container_name|image_id]`

# 容器相关命令

1. 运行容器：

```bash
sudo docker run --gpus all -it -d -rm \
-p <host_port:container_port> \
-v <host_path:container_path> \
-e <var_name=val_value> \
--net=host \
--shm-size=10gb \
--name <container_name> <image_name:image_tag> [/bin/bash]
# --gpus all：允许容器使用所有gpu
# -it：交互模式
# -d：后台运行
# -rm：运行结束删除
# -p：端口映射
# -v：数据卷映射
# -e：环境变量
# --net=host：让容器使用宿主机的网络，而不是创建一个独立的Docker网络。（与-p字段是冲突的）
# --shm-size=10gb：共享内存大小设置为10GB。适用于TensorFlow、PyTorch等需要大共享内存的深度学习任务，否则默认/dev/shm只有64MB，可能导致OOM（内存不足）。
# --name：容器名称
```

要使用`--gpus all`需要额外安装`NVIDIA Container Toolkit`：
```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
```

2. 查看容器：`docker ps`
   1. `-a`：所有容器：`docker ps -a`
   2. `-q`：容器id：`docker ps -q`
   3. `-f`：过滤条件：`docker ps -f “name=<container_name>”`
   4. `-n`：最近创建的n个：`docker ps -n 3`
3. 停止容器：`docker stop [-f] [container_name|container_id]`
   1. `docker stop $(docker ps -q)`：停止所有容器
4. 重启容器：`docker restart <container_name>`
5. 删除容器：`docker rm [-f] [container_name|container_id]`
6. 查看正在运行的docker的日志（若有）：`docker logs <container_name> --tail 10 -f`
   1. `--tail 10`：查看最后10行
   2. `-f`：实时刷新
7. 进入容器：
   1. `docker attach <container_name>`
   2. `docker exec -it <container_name> bash`
   3. `attach`将连接到容器的主进程，可能会影响容器的运行（如使用 Ctrl+C 可能会停止容器）。
   4. `exec`启动一个新的进程，而不是连接到已有的主进程，不会影响容器的主进程，退出新进程不会停止容器。
   5. 也就是说，假如这个容器有启动命令，一直在前台运行某个服务，attach进去之后，其实没办法操作，只能Ctrl+C停止进程，而容器一般都使用-d -rm 之类的命令启动的，这么做就会使容器直接停止并删除。而使用exec进去之后是新开了一个进程，并不会影响主进程的运行，因此很适合进入有启动命令的容器查看相关信息。（总之推荐用exec！！！）
8. 容器提交到镜像：`docker commit <container_name> <newimage_name:tag>`
9. 容器与宿主机文件传输：
   1. 宿主机传输到容器：`docker cp <file_path> <container_id>:<container_file_path>`
   2. 容器传输到宿主机：`docker cp <container_id>:<container_file_path> <file_path>`

# 镜像相关命令

1. 从Dockerfile构建镜像：`docker build -t <image_name:image_tag> [-f </path/to/Dockerfile>] .`
   1. `-t`：指定tag，默认为latest
   2. `-f`：指定Dockerfile文件，没指定默认为当前路径
   3. `.`：决定了docker引擎在构建镜像时可以使用的文件和目录
2. 保存镜像到本地文件：`docker save -o <file_path/file_name.tar> <image_name:tag>`
3. 加载本地保存的镜像：`docker load -i <image_name.tar>`
4. 查看镜像构建过程：`docker history <image_name:tag>`