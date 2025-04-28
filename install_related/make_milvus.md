
- [基础环境配置](#基础环境配置)
- [安装依赖](#安装依赖)
- [下载milvus与第三方库](#下载milvus与第三方库)
- [编译过程中的问题解决](#编译过程中的问题解决)
- [编译后的使用](#编译后的使用)
- [打镜像](#打镜像)


[参开链接1](https://blog.csdn.net/qq_43893755/article/details/135339396)
[参开链接2](https://www.cnblogs.com/AnkleBreaker-ZHX/p/17982844)

# 基础环境配置

**运行容器**

```bash
docker run -it -v /data:/data --name ${container_name} golang:1.22 bash
```

**安装所需工具**

```bash
apt update && apt install -y \
    sudo \
    wget \
    vim \
    ssh \
    git \
    curl \
    lsb-release \
    gnupg \
    cmake \
    make \
    gcc \
    g++ \
    python3 \
    python3-pip \
    libssl-dev \
    libboost-all-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libgtest-dev \
    libopenblas-dev \
    libomp-dev \
    unzip \
    clang \
    clang-format \
    ccache \
    pkg-config \
    libprotobuf-dev \
    protobuf-compiler \
    libcurl4-openssl-dev \
    libevent-dev \
    libunwind-dev \
    libzstd-dev \
    ninja-build
```

速度太慢可以先换源，再执行上面命令

```bash
cat <<EOF > /etc/apt/sources.list
deb https://mirrors.ustc.edu.cn/debian bullseye main contrib non-free
deb https://mirrors.ustc.edu.cn/debian bullseye-updates main contrib non-free
deb https://mirrors.ustc.edu.cn/debian bullseye-backports main contrib non-free
deb https://mirrors.ustc.edu.cn/debian-security bullseye-security main contrib non-free
EOF
```

**安装anaconda并配置环境**

```bash
cd /opt
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /opt/anaconda3
echo "export PATH=/opt/anaconda3/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
conda create -n dev python=3.8
source activate dev
pip install conan==1.59.0
```


# 安装依赖

**安装gvm**

```bash
git clone https://gitcode.net/soulteary/gvm.git
bash gvm/binscripts/gvm-installer
source $HOME/.gvm/scripts/gvm

# 打开.bashrc文件
vim ~/.bashrc
# 在文件末尾追加以下内容
export GO_BINARY_BASE_URL=https://golang.google.cn/dl/
export GOROOT_BOOTSTRAP=$GOROOT
export GO111MODULE=on
export GOPROXY="https://goproxy.cn"
# 使配置生效
source ~/.bashrc
```

**安装fmt**

```bash
cd /root
git clone https://github.com/fmtlib/fmt.git && cd fmt
mkdir _build && cd _build
cmake ..
make -j$(nproc)
sudo make install
```

**安装调试相关依赖**

```bash
apt-get install \
    libunwind8-dev \
    libelf-dev \
    libdwarf-dev
```

**安装folly**

先安装FastFloat：

```bash
cd /root
git clone https://github.com/fastfloat/fast_float.git
```

安装folly：

```bash
cd /root
git clone https://github.com/facebook/folly.git $$ cd folly
mkdir _build && cd _build
cmake .. -DFASTFLOAT_INCLUDE_DIR=/root/fast_float
make -j $(nproc)
make install
```

# 下载milvus与第三方库

**拉取milvus源码**

```bash
cd /root
git clone https://github.com/milvus-io/milvus.git
# 速度太慢可以用gitee
# git clone https://gitee.com/milvus-io/milvus.git
cd milvus
# 切换版本
git checkout v2.5.3
```

**修改部分代码并下载第三方**

```bash
# scripts/install_deps.sh
# clang-format-12 clang-tidy-12 lcov libtool m4 autoconf automake python3 python3-pip
# 修改成：
# clang-format clang-tidy lcov libtool m4 autoconf automake python3 python3-pip

cd /root/milvus
./scripts/install_deps.sh
```

**开始编译**

```bash
cd /root/milvus
make milvus
```

# 编译过程中的问题解决

**问题1**
```bash
conan install: error: unrecognized arguments: --install-folder conan
ERROR: Exiting with code: 2
conan install failed
make: *** [Makefile:251: build-3rdparty] Error 1
```

**解决1**

```bash
# 这是由于conan版本原因
pip uninstall conan
pip install conan==1.60.2
```

**问题2**

```bash
bzip2/1.0.8: WARN: Build folder is dirty, removing it: /root/.conan/data/bzip2/1.0.8/_/_/build/3cfc45772763dad1237052f26c1fe8b2bae3f7d2
bzip2/1.0.8: WARN: Trying to remove corrupted source folder
bzip2/1.0.8: WARN: This can take a while for big packages
bzip2/1.0.8: Configuring sources in /root/.conan/data/bzip2/1.0.8/_/_/source/src
ERROR: bzip2/1.0.8: Error in source() method, line 51
        get(self, **self.conan_data["sources"][self.version], strip_root=True)
        AuthenticationException: 403: Forbidden
conan install failed
make: *** [Makefile:251: build-3rdparty] Error 1
```

**解决2**

这边都是网络导致的下载问题，可以指定直接使用本机的bzip2

```bash
mkdir -p /tmp/bzip2_fake
cd /tmp/bzip2_fake

# 创建conanfile.py文件，内容如下：
# from conans import ConanFile
# class Bzip2SystemFakeConan(ConanFile):
#     name = "bzip2"
#     version = "1.0.8"
#     description = "System-installed bzip2 passthrough"
#     settings = "os", "compiler", "build_type", "arch"

#     def package_info(self):
#         self.cpp_info.includedirs = []  # 如果用系统的，不需要自己加头文件路径
#         self.cpp_info.libdirs = []       # 不指定，默认让系统自己找
#         self.cpp_info.system_libs = ["bz2"]  # ⚡⚡⚡ 这里改成 system_libs

conan remove bzip2/1.0.8 -f
rm -rf ~/.conan/data/bzip2
conan export . bzip2/1.0.8@
```

**问题3**

```bash
boosDownloading boost_1_82_0.tar.bz2:   0%|          | 0.00/116M [00:00<?, ?B/s]boosDownloading boost_1_82_0.tar.bz2:   0%|          | 100k/116M [00:11<3:42:41, 9.0Downloading boost_1_82_0.tar.bz2:   0%|          | 100k/116M [00:28<3:42:41, 9.0Downloading boost_1_82_0.tar.bz2:   0%|          | 200k/116M [00:52<9:29:30, 3.5Downloading boost_1_82_0.tar.bz2:   0%|          | 200k/116M [01:08<9:29:30, 3.54kB/s]boost/1.82.0: WARN: Could not download from the URL https://sourceforge.net/projects/boost/files/boost/1.82.0/boost_1_82_0.tar.bz2: Download failed, check server, possibly try again
HTTPSConnectionPool(host='zenlayer.dl.sourceforge.net', port=443): Read timed out.. Trying another mirror.
ERROR: boost/1.82.0: Error in source() method, line 810
        get(self, **self.conan_data["sources"][self.version],
        ConanException: All downloads from (2) URLs have failed.
conan install failed
make: *** [Makefile:251: build-3rdparty] Error 1
```

**解决3**

也是网络原因导致的下载错误。提前下载。

```bash
# 提前下载，并保存到/root/tmp/boost_1_82_0.tar.bz2
# 编译文件 ~/.conan/data/boost/1.82.0/_/_/export/conandata.yml
# 将对应版本做如下更改：
"1.82.0":
    url:
      - "file:///root/tmp/boost_1_82_0.tar.bz2"
    sha256: "a6e1ab9b0860e6a2881dd7b21fe9f737a095e5f33a3a874afc6a345228597ee6"
```


**问题4**

```bash
./bin/milvus: error while loading shared libraries: libfolly_exception_tracer_base.so.0.58.0-dev: cannot open shared object file: No such file or directory
```

**解决4**

只要安装了上面的所有依赖，这边一定是有的，只要设置一下环境变量即可

```bash
# 制作软连接
sudo ln -s /root/milvus/cmake_build/lib/libfolly_exception_tracer_base.so.0.58.0-dev /usr/local/lib/libfolly_exception_tracer_base.so.0.58.0-dev
# 配置环境变量，在~/.bashrc中添加：
export PATH=$PATH:"$HOME/.rustup"
export PATH="$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY:/usr/local/lib:/root/milvus/cmake_build/azure
```


# 编译后的使用

**启动etcd**

```bash
cd /root/tmp
# 下载 etcd（选择与 Milvus 兼容的版本，如 v3.5.0）
wget https://github.com/etcd-io/etcd/releases/download/v3.5.0/etcd-v3.5.0-linux-amd64.tar.gz
tar -xvf etcd-v3.5.0-linux-amd64.tar.gz
cd etcd-v3.5.0-linux-amd64
# 启动 etcd
./etcd --data-dir=/tmp/etcd-data &
```

可以添加到环境变量：

```bash
# 在 ~/.bashrc 中添加
export PATH=$PATH:/root/tmp/etcd-v3.5.0-linux-amd64/
```

milvus.yaml中相应配置如下：

```yaml
etcd:
  useEmbedEtcd: false
  endpoints: ["localhost:2379"]
```

启动成功后，可以查看注册情况

```bash
[root@0748817ab3fc:/root/milvus]
$ etcdctl --endpoints=http://127.0.0.1:2379 get --prefix by-dev/meta/session/ --keys-only
# 也可以把 --keys-only 去掉，查看端口有没有正确
by-dev/meta/session/datacoord       # DataCoord 节点
by-dev/meta/session/datanode-8      # DataNode 节点（ID 是 8）
by-dev/meta/session/id              # session ID 分配器
by-dev/meta/session/indexcoord      # IndexCoord 节点
by-dev/meta/session/indexnode-8     # IndexNode 节点
by-dev/meta/session/proxy-8         # Proxy 节点
by-dev/meta/session/querycoord      # QueryCoord 节点
by-dev/meta/session/querynode-8     # QueryNode 节点
by-dev/meta/session/rootcoord       # RootCoord 节点
```

这代表 Milvus 服务端组件基本都起来了，它们都能成功注册到 etcd，说明启动整体成功了一大半！也可以使用curl查看：
```bash
curl http://127.0.0.1:2379/health
# 会显示：{"health":"true","reason":""}
```

**启动MinIO**

先配置minio的密钥：

```bash
# ~/.bashrc 中添加（跟下面yaml的内容对应）：
export MINIO_ROOT_USER="minioadmin"
export MINIO_ROOT_PASSWORD="minioadmin"
```

然后下载minio并运行：

```bash
cd /root
# 下载并启动 MinIO（如果尚未安装）
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
./minio server /tmp/minio-data --console-address ":9001" &
```

milvus.yaml中相应配置如下：

```bash
minio:
  address: localhost
  port: 9000
  accessKeyID: minioadmin
  secretAccessKey: minioadmin
  bucketName: milvus-bucket
  useSSL: false
  enable: true  # 必须启用
```

下载mc进行验证配置：

```bash
cd /root
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
mv mc /usr/local/bin
```

然后运行以下命令，这会将 Minio 服务配置为 myminio，并使用 minioadmin 作为用户名和密码。

```bash
[root@0748817ab3fc:/root]
$ mc alias set myminio http://localhost:9000 minioadmin minioadmin
mc: Configuration written to `/root/.mc/config.json`. Please update your access credentials.
mc: Successfully created `/root/.mc/share`.
mc: Initialized share uploads `/root/.mc/share/uploads.json` file.
mc: Initialized share downloads `/root/.mc/share/downloads.json` file.
Added `myminio` successfully.
[root@0748817ab3fc:/root]
$ mc mb myminio/milvus-bucket
Bucket created successfully `myminio/milvus-bucket`.
[root@0748817ab3fc:/root]
$ mc ls myminio
[2025-04-28 08:00:52 UTC]     0B a-bucket/
[2025-04-28 08:44:24 UTC]     0B milvus-bucket/
```


**运行milvus的standalone**

```bash
./bin/milvus run standalone
```

运行成功后，进程信息如下：

```bash
[root@0748817ab3fc:/root/milvus]
$ ps -ef
UID        PID  PPID  C STIME TTY          TIME CMD
root         1     0  0 02:45 pts/0    00:00:00 bash
root       161     0  0 02:45 pts/1    00:00:00 bash
root      1756     0  0 06:09 pts/2    00:00:00 bash
root      3609  1756  0 07:41 pts/2    00:00:07 ./etcd --data-dir=/tmp/etcd-data
root      5004  1756  0 07:55 pts/2    00:00:02 ./minio server /tmp/minio-data --console-address :9001
root      5667     0  0 08:02 pts/3    00:00:00 bash
root      6725  1756 45 08:24 pts/2    00:00:07 ./bin/milvus run standalone
root      6802     1  0 08:24 ?        00:00:00 /opt/anaconda3/bin/dbus-daemon --syslog --fork --print-pid 4 --print-address 6 --session
root      6901  5667  0 08:24 pts/3    00:00:00 ps -ef
```


# 打镜像

```bash
docker commit -a "damonzheng" -m "make milvus" damonzheng_milvus milvus_maked:latest
```

没有磁盘空间，所以没有 commit 成功