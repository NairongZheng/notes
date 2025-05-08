- [linux操作相关](#linux操作相关)
  - [Linux目录结构](#linux目录结构)
  - [安装与设置](#安装与设置)
  - [查看信息](#查看信息)
  - [文件操作](#文件操作)
  - [网络端口操作](#网络端口操作)
  - [screen操作](#screen操作)
  - [其他操作](#其他操作)
  - [文本处理命令](#文本处理命令)
  - [Linux用户和组管理](#linux用户和组管理)
    - [用户管理](#用户管理)
    - [组管理](#组管理)
    - [权限管理](#权限管理)
- [windows操作相关](#windows操作相关)
  - [windows目录结构](#windows目录结构)
  - [查看信息](#查看信息-1)
  - [网络端口操作](#网络端口操作-1)
  - [其他操作](#其他操作-1)
- [环境相关](#环境相关)
- [问题](#问题)


# linux操作相关

## Linux目录结构

```plaintext
/                       # 根目录，所有文件和目录的起始点
├── bin/                # 基本二进制命令，所有用户都可用的工具（如 ls, cp, mv 等）
├── boot/               # 启动文件，包含内核和启动加载程序的配置文件（如 vmlinuz, grub）
├── dev/                # 设备文件，代表硬件设备（如 /dev/sda, /dev/null）
├── etc/                # 系统配置文件，存储系统的配置文件（如 /etc/passwd, /etc/fstab）
├── home/               # 用户家目录，每个用户的个人文件（如 /home/username）
├── lib/                # 系统共享库文件，供程序运行时使用（如 libc.so, libm.so）
├── lost+found/         # 文件系统恢复目录，用于存放 fsck 恢复的文件
├── media/              # 可移动设备挂载点（如 USB, CD/DVD 挂载）
├── mnt/                # 临时挂载点，用于系统管理员挂载外部文件系统
├── opt/                # 第三方应用程序，通常为不依赖包管理器的安装软件
├── proc/               # 虚拟文件系统，包含内核和进程信息（如 /proc/cpuinfo, /proc/[pid]）
├── root/               # root 用户的家目录，系统管理员的个人文件
├── run/                # 运行时数据，存储系统当前会话的临时文件（如 PID 文件、锁文件）
├── sbin/               # 系统管理命令，供管理员（root 用户）使用（如 fsck, shutdown）
├── srv/                # 服务数据，存储由系统提供的服务的数据（如 /srv/ftp, /srv/http）
├── sys/                # 虚拟文件系统，提供系统硬件信息和内核参数
├── tmp/                # 临时文件，存储临时数据，系统重启时会清空
├── usr/                # 用户级程序和数据，包含大部分应用程序和共享文件
│   ├── bin/            # 用户命令二进制文件，会被包管理器（apt、dnf、yum等）管理的用户程序（如 /usr/bin/ls, /usr/bin/gcc）
│   ├── lib/            # 用户程序的库文件（如 /usr/lib/libc.so）
│   ├── share/          # 共享文件，如文档、图标等（如 /usr/share/man）
│   ├── local/          # 本地安装的软件目录，不由系统包管理器如apt等控制（如源码安装的软件）
│   │   ├── bin/        # 用户手动编译/安装的可执行文件目录，优先级通常高于 /usr/bin（如 /usr/local/bin/myapp）
│   │   ├── lib/        # 用户手动安装程序的库文件（如 /usr/local/lib/libmylib.so）
├── var/                # 可变数据，存储日志、缓存、邮件队列等数据
│   ├── log/            # 日志文件（如 /var/log/syslog）
│   ├── spool/          # 打印队列或邮件队列（如 /var/spool/mail）
│   └── cache/          # 应用程序缓存（如 /var/cache/apt）
```

## 安装与设置

**安装**

```bash
apt install -y curl                 # curl
apt install -y wget                 # wget
apt install -y vim                  # vim
apt install -y git                  # git
apt install -y ssh                  # ssh
apt install -y screen               # screen
apt install -y htop                 # htop
apt install -y lsof                 # lsof
apt install -y netcat               # netcat(nc)
apt install -y telnet               # telnet
apt install -y iputils-ping         # ping等工具
apt install -y net-tools            # ifconfig、netstat等
apt install -y iproute2             # 网络工具如ip（iproute2已取代了net-tools）
apt install -y cloc                 # 统计代码量的工具
```

**配置**

`~/.bashrc`部分配置：

```bash
# 设置终端提示符样式
PS1="\n\e[1;37m[\e[m\e[1;32m\u\e[m\e[1;33m@\e[m\e[1;35m\h\e[m:\e[m\$PWD\e[m\e[1;37m]\e[m\e[1;36m\e[m\n$ "
# 禁止分页显示
export PAGER=cat
export MANPAGER=cat
export GIT_PAGER=cat
# 设置编码
export LANG=C.UTF-8 # zh_CN.UTF-8为中文，en_US.UTF-8为英文
export LC_ALL=C.UTF-8
```

**环境变量语法**

```bash
# export PATH=$PATH:<new_path>    # 无引号，带空格路径会出错
# export PATH=$PATH:"<new_path>"  # 没错，但是自定义的在最后，使用的时候还是用到了前面的版本
export PATH="<new_path>:$PATH"    # 推荐将自定义工具路径加到 PATH 前面
```

## 查看信息

**查看系统相关信息**

```bash
# 1. 查看linux版本等信息：cat /etc/os-release
# 2. 查看当前使用的shell：echo $SHELL 或者 echo $0
# 3. 查看当前shell的PID：echo $$
# 4. 查看内存：free -h
# 5. 查看系统系统区域设置（语言、字符编码、日期格式等）：locale
# 6. 查看系统支持语言：locale -a
# 7. 查看系统时间：date
```

**查看文件系统的磁盘空间使用情况 df (Disk Free)**

```bash
df -h
    # -h：以人类可读的格式显示（GB、MB、KB）。
    # -T：显示文件系统的类型。
    # -a：显示所有文件系统，包括那些 0 字节的。
    # -l：只显示本地文件系统，不包括网络挂载的。
    # -i：显示 inode 使用情况，而不是磁盘空间。
df -h ${file_path}
    # 可以直接查看这个文件/目录属于哪个空间
```

**查看文件或目录占用的磁盘空间 du (Disk Usage)**

```bash
du -sh ./*
    # -h：以人类可读的格式显示。
    # -s：只显示每个文件或目录的总计，而不列出子目录。
    # -a：显示所有文件和目录的大小，而不是仅显示目录。
    # -c：显示总和。
    # -d N：显示 N 层目录的大小。
    # --max-depth=N：限制递归的深度，N 是递归的层数。
    # -x：只计算同一个文件系统中的文件，不跨文件系统。
```

**查看目录结构**

```bash
tree
    # -L n：限制显示的目录层级为 n 级。例如：tree -L 2
    # -d：只显示目录，不显示文件
    # -a：显示所有文件和目录，包括隐藏文件
    # -f：显示完整路径（从根目录开始的路径）。
```

**查看代码量**

```bash
cloc <proj_path>

# 显示类似如下（语言/文件数/空行数/注释行数/实际代码行数）：
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
JSON                            86            986              0         343291
Python                         180           6545           9527          87289
HTML                             1            749             10           4117
Markdown                         4            199              0           3215
C++                              4            176             54            850
Protocol Buffers                 9            126             67            558
C                                1            114             58            499
CMake                            9             64             26            334
Bourne Shell                    17             37             35            127
C/C++ Header                     2             20              0            118
make                             1             54             50            101
DOS Batch                        1              9              3             12
INI                              2              2              0              5
-------------------------------------------------------------------------------
SUM:                           317           9081           9830         440516
-------------------------------------------------------------------------------
```

## 文件操作

**文件解压**

```bash
# 1. zip文件解压：unzip <filename>
# 2. tar文件解压：tar -xvf <filename>
# 3. tar.gz文件解压：tar -zxvf <filename>
# 4. rar文件解压：unrar x <filename>
# 5. 7z文件解压：7za x <filename>
```

**文件复制**

```bash
# 复制文件夹并排除特定子文件夹
rsync -av --exclude="<subfolder1>" --exclude="<subfolder2>" <source_dir> <destination_dir>
    # -a：归档模式，表示递归复制并保持文件属性
    # -v：详细模式，显示复制过程中的详细信息
# 如：rsync -av --exclude="*.T" --exclude="znr" --exclude=".git" ../up/bwy_v4/bwy ./
```

**文件传输scp**

具体查看[ssh配置部分](./ssh.md/#scp服务器间拷贝文件)

```bash
# 本地传到服务器：
scp -P <port> <filepath_windows> root@<remote_ip>:<filepath_linux>
# 服务器传到本地：
scp -P <port> root@<remote_ip>:<filepath_linux> <filepath_windows>
```

**文件传输sz/rz**

以mobaXterm为例：

```bash
# 1. 本地上传到服务器：`rz` && `ctrl + 鼠标右键` && `Send file using Z-modem` && `选择文件`
# 2. 服务器下载到本地：`sz filename` && `ctrl + 鼠标右键` && `Receive file using Z-modem`
# 3. 中途取消操作：`ctrl + x`按4到5次
```

**文件校验**

```bash
md5sum <filename>       # md5
sha256sum <filename>    # sha256
```

**文件加密**

具体查看[gpg文件加密](./gpg.md)

```bash
# 1. 安装gpg：`sudo apt install gnupg`
# 2. 加密文件：`gpg -c ${filename}`，输入两次密码，生成`${filename.gpg}`
# 3. 解密文件：`gpg -d ${filename.gpg} > ${filename}`
```

**查找文件**

```bash
find ${base_path} -name ${file_name} [options]
    # -name：按文件名匹配查找
    # -size：查找特定大小的文件。例如，-size +10M 查找大于 10MB 的文件，-size -10M 查找小于 10MB 的文件
    # -iname：不区分大小写进行查找
    # -type：按照文件类型查找。
        # -type f：文件
        # -type d：目录
        # -type l：符号链接文件
    # -perm：查找特定权限文件，如 -perm 755
    # -user：查找指定用户文件
    # -ls：列出详细信息
```

## 网络端口操作

**查看网络信息**

```bash
ip [OPTIONS] addr      # 查看ip地址信息
ip [OPTIONS] link      # 查看网卡信息
    # -br[ief]：简洁版
```

每种 IP/网卡通常代表什么：

| 接口名                    | 作用说明                                                       |
| ------------------------- | -------------------------------------------------------------- |
| `lo`                      | 本地回环接口，127.0.0.1，供本机自己通信用。                    |
| `eth0`, `ens33`, `ens160` | 有线网卡（物理或虚拟机的），通常连接局域网/互联网。            |
| `wlan0`                   | 无线网卡（Wi-Fi）。                                            |
| `docker0`                 | Docker 默认创建的网桥网络，容器用来和宿主机通信。              |
| `br-xxxxxxx`              | 用户创建的 Docker 自定义 bridge 网络。                         |
| `vethxxxxx`               | Docker 容器内部的虚拟网卡，一般是 veth 对，连接容器和 bridge。 |
| `virbr0`                  | KVM 虚拟机系统使用的虚拟桥接网卡。                             |
| `tun0`, `tap0`            | VPN 创建的虚拟网络接口，比如 OpenVPN。                         |


**查看端口使用情况**

```bash
netstat -tunlp
    # -t：(tcp) 仅显示tcp相关选项
    # -u：(udp)仅显示udp相关选项
    # -n：拒绝显示别名，能显示数字的全部转化为数字
    # -l：仅列出在Listen(监听)的服务状态
    # -p：显示建立相关链接的程序名
```

**查看端口监听状态**

```bash
# 查看端口监听状态
lsof -i -P -n | grep LISTEN
    # -i：显示与网络相关的文件
    # -P：显示端口号而不是服务名（ssh就是22之类的）
    # -n：不解析主机名（localhost就显示127.0.0.1之类的）

# 查看具体某端口状态
lsof -i:<port>
```

**其他**

```bash
iptables -L             # 查看防火墙规则
telnet <ip> <port>      # 测试能否连上某<ip:port>
```

## screen操作
1. 创建screen：`screen -S <screen_name>`
2. 退出screen：`Ctrl+a+d`
3. 查看screen：`screen -ls`
4. 查看是否在screen中：`echo $STY`，非空则表示在screen中
5. 进入已有screen：`screen -r <screen_name>`
6. 断开并保留screen：`screen -d <screen_name>`，若要强制进入别人正在查看的screen可以先用这个命令断开
7. 删除screen：`kill -9 <screen_pid>` && `screen -wipe`
8. 关闭所有会话：`screen -X quit`
9. 配置`~/.screenrc`（screen屏闪、鼠标滚轮、语言支持等）：

```bash
# 在~/.screenrc中添加以下内容，然后关闭所有会话再重启
# 避免闪屏
vbell off
defscrollback 10000
# 启用鼠标支持
termcapinfo xterm* ti@:te@
mousetrack on
# 设置编码为UTF-8
defutf8 on
```

## 其他操作

1. [重定向输出到黑洞](https://blog.csdn.net/longgeaisisi/article/details/90519690)：`/dev/null 2>&1`

## 文本处理命令

**查看文件内容**

```bash
# cat：显示文件内容，-n带行号
cat [-n] <filename>
# head：查看文件前n行（默认前10行）
head [-n] <filename>
# tail：查看文件后n行
tail [-n] <filename>
tail [-f] <filename> # 实时查看文件内容（适用于不断刷新的日志文件）
# less：分页查看文件，空格可以翻页（非常非常好用）
less <filename>
```

**grep查找文本行**

（[正则表达式](./shell.md/#regexp)）

```bash
grep [OPTIONS] "<keyword>" <filename>
    # -i：忽略大小写查找
    # -n：显示行号
    # -r：递归查找某目录下的文件
    # -l：只显示匹配的文件名，不显示内容
    # 还可以匹配正则表达式 ``
    # -E：支持高阶正则表达式
```

**sed文本流编辑**

```bash
sed [-i] 's/<old_content>/<new_content>/g' <filename>
    # -i：直接修改文件（没有-i则仅显示结果，不修改文件）
    # s：替换文本（默认替换每行第一个匹配）
    # g：行内全面替换
sed [-i] '/<content>/d' <filename>
    # -i：直接修改文件（没有-i则仅显示结果，不修改文件）
    # d：删除行
```

**awk处理结构化文本**

```bash
# 按列提取数据（空格或制表符分隔）
awk '{print $1}' <filename>  # 显示第一列
awk '{print $1, $3}' <filename>  # 显示第一列和第三列
# 查找包含 "<content>" 的行，并显示第二列
awk '/<content>/ {print $2}' <filename>
```

**cut按列提取文本**

```bash
cut -d ":" -f1,3 <filename>
    # -d：使用":"作为分隔符
    # -f：指定提取列数，第一列与第三列
```

**wc统计文件内容**

```bash
# 统计行数、单词数、字符数
wc [OPTIONS] <filename>
    # -l：仅统计行数
    # -w：仅统计单词数
    # -c：仅统计字符数
```

**tee将输出同时写入文件和终端**

```bash
# 将命令结果保存到文件，同时显示在终端
ls -l | tee <save_filename>
```

**综合使用案例**

```bash
# kill包含<content>的进程
ps aux | grep <content> | awk '{print $2}' | xargs kill -9
    # ps aux：获取所有进程
    # grep <content>：获取包含<content>的进程
    # awk '{print $2}'：获取这些进程的pid
    # xargs kill -9：将这些进程的pid作为kill -9的参数
```

## Linux用户和组管理

**文件介绍**

Linux的用户和组信息存储在以下文件中：
```bash
# /etc/passwd：存储所有用户的信息（用户名、UID、GID、主目录、Shell）。
    # 如damonzheng:x:1003:1003:damonzheng:/home/damonzheng:/bin/bash
    # damonzheng：用户名
    # x：密码占位符（真实密码存储在/etc/shadow）
    # 1003：用户ID（UID）
    # 1003：组ID（GID）
    # damonzheng：用户描述
    # /home/damonzheng：用户主目录
    # /bin/bash：用户默认shell
# /etc/shadow：存储用户的加密密码及账户安全信息（只有root能访问）。
# /etc/group：存储所有组的信息（组名、GID、组成员）。
    # 如sudo:x:27:ubuntu
    # sudo：组名
    # x：密码占位符（通常不使用）
    # 27：组ID（GID）
    # ubuntu：组成员（多个成员用,分隔）
# /etc/gshadow：存储组的加密密码和安全信息。
```

### 用户管理

**新建用户**
```bash
# 创建用户（不创建主目录）
sudo useradd <user_name>
# 创建用户，并自动创建主目录
sudo useradd -m <user_name>
# 创建用户，并自动创建主目录，并指定默认shell为bash
sudo useradd -m -s /bin/bash <user_name>
# 创建用户，指定 UID 和 GID
sudo useradd -u <uid> -g <gid> <user_name>
```

**删除用户**
```bash
# 删除用户，但保留用户的主目录
sudo userdel <user_name>
# 删除用户，并同时删除其主目录
sudo userdel -r <user_name>
```

**修改用户**
```bash
# 修改用户名
sudo usermod -l <new_user_name> <old_user_name>
# 修改用户 UID
sudo usermod -u <uid> <user_name>
# 修改用户的主目录（同时移动文件）
sudo usermod -d /new/home/path -m <user_name>
# 修改用户默认 shell
sudo usermod -s /bin/zsh <user_name>
# 将用户添加到附加组（不影响主组）
sudo usermod -aG <group_name> <user_name>
# 修改用户的主组
sudo usermod -g <group_name> <user_name>
# 将<username>只加入<group1>, <group2>, <group3>组，并移除其他附加组（主组不变）。
sudo usermod -G <group1>,<group2>,<group3> <username>
```

**查看用户信息**
```bash
# 查询当前用户
whoami
# 查询所有已登录用户
who
# 查询当前活动用户
w
# 查看某个用户的详细信息
id <user_name>
# 查看系统所有用户（读取/etc/passwd文件）
cat /etc/passwd | cut -d: -f1
```

**切换用户**
```bash
# 以另一个用户的身份执行命令
su - <user_name>
```

**设置用户无需输入sudo密码**
```bash
# 按照以下顺序操作
# sudo visudo
# <%sudo ALL=(ALL:ALL) ALL>下面添加<your_username> ALL=(ALL) NOPASSWD:ALL
# ctrl + x 保存再回车就退出
# 或者直接用命令：sudo echo "<user_name> ALL=(ALL:ALL) NOPASSWD: ALL" >>/etc/sudoers
```

### 组管理

**创建组**
```bash
# 添加新组
sudo groupadd <group_name>
```

**删除组**
```bash
# 删除一个组
sudo groupdel <group_name>
```

**从组中移除用户**
```bash
# 从<group_name>组中 移除<user_name>用户。不会删除用户，只是移除其组成员身份。
sudo gpasswd -d <user_name> <group_name>
```

**查询组信息**
```bash
# 显示当前用户所属的所有组
groups
# 显示指定用户所属的所有组
groups <user_name>
# 显示组的详细信息
getent group <group_name>
# 查看某个组的所有用户
grep <group_name> /etc/group
```

### 权限管理

**文件权限基础**
```bash
drwxr-xr-x  2 user group  4096 Mar 10 06:28 example/
# d：文件类型（d 表示目录，- 表示普通文件）
# rwx：所有者权限（读、写、执行）
# r-x：所属组权限（读、执行）
# r-x：其他用户权限（读、执行）
# 2：硬链接数
# user：该目录的所有者
# group：该目录的所属组
```

**修改文件/目录权限**
```bash
# 修改文件权限（数字方式）
chmod [-R] 755 <file_or_dir>
# 修改文件权限（符号方式）
chmod [-R] u+rwx,g+rx,o+rx <file_or_dir>
```

**修改文件/目录的所有者**
```bash
# 修改文件的所有者
sudo chown [-R] <new_owner> <file_or_dir>
# 修改文件的所属组
sudo chown [-R] :<new_group> <file_or_dir>
# 修改文件的所有者和所属组
sudo chown [-R] <new_owner>:<new_group> <file_or_dir>
```

# windows操作相关

## windows目录结构

```plaintext
C:\                                 # 系统盘（默认），存储 Windows 操作系统和应用程序
├── Program Files\                  # 默认安装目录，存放 64 位应用程序（如 C:\Program Files\Google\Chrome）
├── Program Files (x86)\            # 存放 32 位应用程序（仅在 64 位 Windows 上存在）
├── Windows\                        # 操作系统核心目录，包含 Windows 组件、配置和驱动
│   ├── System32\                   # 64 位系统核心文件，存储 DLL、EXE、驱动程序（如 cmd.exe, notepad.exe）
│   ├── SysWOW64\                   # 32 位系统核心文件，仅在 64 位 Windows 上存在
│   ├── WinSxS\                     # Windows 组件存储，管理不同版本的 DLL 以防止冲突
│   ├── Temp\                       # 系统临时文件，程序运行时的缓存文件
│   ├── Fonts\                      # 存放系统字体文件（如 Arial.ttf, SimSun.ttc）
│   ├── Resources\                  # 主题文件、界面资源（如壁纸、音效）
│   ├── Logs\                       # 系统日志文件
│   ├── INF\                        # 驱动安装信息文件
│   ├── Tasks\                      # 计划任务存储目录
│   ├── SoftwareDistribution\       # Windows 更新文件缓存
│   └── Web\                        # Edge 浏览器相关文件和壁纸存储
├── Users\                          # 用户目录，存放用户的个人数据
│   ├── Administrator\              # 管理员账户目录
│   ├── Default\                    # 默认用户配置模板，新建用户时会复制该目录
│   ├── Public\                     # 共享文件夹，所有用户可访问
│   ├── [用户名]\                    # 个人用户目录
│   │   ├── Desktop\                # 桌面文件
│   │   ├── Documents\              # 文档目录
│   │   ├── Downloads\              # 下载目录
│   │   ├── Pictures\               # 图片目录
│   │   ├── Videos\                 # 视频目录
│   │   ├── Music\                  # 音乐目录
│   │   ├── AppData\                # 应用程序数据（用户级）
│   │   │   ├── Local\              # 本地应用数据（如缓存文件）
│   │   │   ├── LocalLow\           # 低权限应用数据
│   │   │   └── Roaming\            # 可同步的用户数据（网络账户时可同步）
├── ProgramData\                    # 公共应用数据（类似 Linux 的 `/var` 目录）
├── PerfLogs\                       # 性能日志文件
├── Recovery\                       # 系统恢复文件
├── System Volume Information\      # 备份和恢复点信息（普通用户无法访问）
├── Recycle Bin\                    # 回收站，存放被删除但未永久清除的文件
└── Temp\                           # 系统级临时文件（可与 `C:\Windows\Temp` 不同）
```

## 查看信息

1. 查看系统详细信息：`systeminfo`

## 网络端口操作

1. 查看本机ip：`ipconfig`

## 其他操作

1. 开启WSL功能：管理员模式在`PowerShell`中运行`dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`

# 环境相关

1. [python项目自动生成环境配置文件requirements.txt](https://blog.csdn.net/pearl8899/article/details/113877334)：`pipreqs .`
2. 导出conda环境配置：`conda env export > environment.yml`
3. [mobaXterm使用本地conda](https://www.cnblogs.com/AnonymousDestroyer/p/17258702.html)：在`~/.bashrc`中添加以下代码：

```bash
export PATH=/drives/d/app/anaconda/install/Scripts:$PATH
export PYTHONIOENCODING=utf-8
if [[ "${OSTYPE}" == 'cygwin' ]]; then
    set -o igncr
    export SHELLOPTS
fi
```

# 问题

1. [linux中按上下左右键为什么变成\^\[\[A\^\[\[B\^\[\[C\^\[\[D](https://www.zhihu.com/question/31429658)：输入`bash`解决