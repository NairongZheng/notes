
- [其他命令](#其他命令)
- [容器相关命令](#容器相关命令)
- [镜像相关命令](#镜像相关命令)
- [volume相关](#volume相关)
- [network相关](#network相关)
- [其他](#其他)
  - [端口转发](#端口转发)


# 其他命令
1. 启动docker服务：`systemctl start docker`（老版用`service docker start`）
2. 重启docker服务：`systemctl restart docker`（老版用`service docker restart`）
3. 停止docker服务：`systemctl stop docker`（老版用`service docker stop`）
4. 检查docker状态：`systemctl status docker`（老版用`service docker status`）
5. 设置docker开机自启：`systemctl enable docker`
6. 取消docker开机自启：`systemctl disenable docker`
7. 查看docker磁盘使用情况：`docker system df [-v]`
8. 清理build缓存：`docker builder prune`
9. 查看容器/镜像的详细信息：`docker inspect [container_name|image_id]`

# 容器相关命令

1. **运行容器**：

```bash
sudo docker run --gpus all -it -d -rm \
                -p <host_port:container_port> \
                -v <host_path:container_path> \
                -e <var_name=val_value> \
                --net=host \
                --network <network_name> \
                --ip <custom_ip> \
                --shm-size=10gb \
                --name <container_name> <image_name:image_tag> [/bin/bash]
# --gpus all：允许容器使用所有gpu
# -it：交互模式
# -d：后台运行
# -rm：运行结束删除
# -p：端口映射（有-p就不能有--net=host）
# -v：数据卷映射
# -e：环境变量
# --net=host：容器直接使用宿主机的网络。（与-p、--network等字段是冲突）
# --network：采用哪个docker网络，允许同一个网络间的容器互相通信
# --ip：固定ip（必须跟--network配合使用）
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

2. **查看容器**：`docker ps`
   1. `-a`：所有容器：`docker ps -a`
   2. `-q`：容器id：`docker ps -q`
   3. `-f`：过滤条件：`docker ps -f “name=<container_name>”`
   4. `-n`：最近创建的n个：`docker ps -n 3`
3. **查看容器占用资源**：`docker stats [--no-stream] [container_name|container_id]`
   1. `--no-stream`：一次性快照，不实时刷新
4. **停止容器**：`docker stop [-f] [container_name|container_id]`
   1. `docker stop $(docker ps -q)`：停止所有容器
5. **重启容器**：`docker restart <container_name>`
6. **删除容器**：`docker rm [-f] [container_name|container_id]`
7. **暂停容器**：`docker pause [container_name|container_id]`（短暂释放资源时使用）
8. **取消暂停**：`docker unpause [container_name|container_id]`
9.  **查看正在运行的docker的日志（若有）**：`docker logs <container_name> --tail 10 -f`
   1. `--tail 10`：查看最后10行
   2. `-f`：实时刷新
10. **进入容器**：
   1. `docker attach <container_name>`
   2. `docker exec -it <container_name> bash`
   3. `attach`将连接到容器的主进程，可能会影响容器的运行（如使用 Ctrl+C 可能会停止容器）。
   4. `exec`启动一个新的进程，而不是连接到已有的主进程，不会影响容器的主进程，退出新进程不会停止容器。
   5. 也就是说，假如这个容器有启动命令，一直在前台运行某个服务，attach进去之后，其实没办法操作，只能Ctrl+C停止进程，而容器一般都使用-d -rm 之类的命令启动的，这么做就会使容器直接停止并删除。而使用exec进去之后是新开了一个进程，并不会影响主进程的运行，因此很适合进入有启动命令的容器查看相关信息。（总之推荐用exec！！！）
11. **容器与宿主机文件传输**：
    1.  宿主机传输到容器：`docker cp <file_path> <container_id>:<container_file_path>`
    2.  容器传输到宿主机：`docker cp <container_id>:<container_file_path> <file_path>`
12. **容器提交到镜像**：`docker commit <container_name> <newimage_name:tag>`，可以有以下参数：
```bash
docker commit -a "damonzheng" \
              -m "update base image with new configurations" \
              --change 'CMD ["bash", "start.sh"]' \
              --change "EXPOSE 80" \
              --change "ENV APP_ENV=production" \
              <container_id/container_name> <newimage_name:tag>

# -a, --author：指定作者
# -m, --message：添加注释
# --change：修改Docker配置
    # 修改CMD：--change 'CMD ["bash", "start.sh"]'
    # 修改ENTRYPOINT：--change 'ENTRYPOINT ["python", "app.py"]'
    # 修改环境变量ENV：--change "ENV APP_ENV=production"
    # 修改工作目录WORKDIR：--change "WORKDIR /data/code"
    # 修改暴露端口EXPOSE：--change "EXPOSE 8080"
```


# 镜像相关命令

1. **从Dockerfile构建镜像**：`docker build -t <image_name:image_tag> [-f </path/to/Dockerfile>] .`
   1. `-t`：指定tag，默认为latest
   2. `-f`：指定Dockerfile文件，没指定默认为当前路径
   3. `.`：决定了docker引擎在构建镜像时可以使用的文件和目录
2. **保存镜像到本地文件**：`docker save -o <file_path/file_name.tar> <image_name:tag>`
3. **加载本地保存的镜像**：`docker load -i <image_name.tar>`
4. **查看镜像构建过程**：`docker history <image_name:tag>`


# volume相关

1. **创建命名volume**：`docker volume create <volume_name>`
2. **查看所有volume**：`docker volume ls`
3. **查看volume详细信息**：`docker volume inspect <volume_name>`
4. **删除volume**：`docker volume rm <volume_name>`（正在使用则无法删除）


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

1. **创建network**：`docker network create <network_name>`
2. **查看所有network**：`docker network ls`
3. **查看network详细信息**：`docker network inspect <network_name>`
4. **将容器连接到network**：`docker network connect <network_name> <container_name_or_id>`
5. **将容器从指定network断开**：`docker network disconnect <network_name> <container_name_or_id>`
6. **删除network**：`docker network rm <network_name>`


使用docker network的好处：

1. 容器自动内网互通：同一个 network 下的容器，自动能用 容器名当域名互相访问，不用记 IP 地址
2. 自定义IP段：自定义网络时可以指定 IP 段、子网掩码，方便跟现有基础设施融合
3. 隔离性更强：不同的 network 之间 默认互相隔离，安全性高，防止容器乱串
4. 支持多网卡容器：一个容器可以连到多个 network，相当于多块网卡，复杂应用（如代理服务器）必备
5. 跨主机扩展：使用 Swarm / Kubernetes，可以把 network 跨宿主机打通，做跨机集群
6. 内建负载均衡：多个容器用同一个名字（服务名）注册时，docker network 会自动轮询负载均衡请求（bridge overlay支持）

# 其他

## 端口转发

有时候忘记做端口映射，可以临时使用 `socat` 端口转发的方式：

```bash
# 基本语法：
socat <source> <destination>
    # <source>: 监听的源端口或文件、设备等。
    # <destination>: 要转发到的目标端口或文件、设备等。

# 将 <container_ip>:<container_port> 转发到宿主机的 <dev_port>
socat TCP-LISTEN:<dev_port>,reuseaddr,fork TCP:<container_ip>:<container_port>
# 监听宿主机的 60010 端口，并将请求转发到容器 172.17.0.2 的 3000 端口
socat TCP-LISTEN:60010,reuseaddr,fork TCP:172.17.0.2:3000
```