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
/               # 根目录，所有文件和目录的起始点
├── bin/        # 基本二进制命令，所有用户都可用的工具（如 ls, cp, mv 等）
├── boot/       # 启动文件，包含内核和启动加载程序的配置文件（如 vmlinuz, grub）
├── dev/        # 设备文件，代表硬件设备（如 /dev/sda, /dev/null）
├── etc/        # 系统配置文件，存储系统的配置文件（如 /etc/passwd, /etc/fstab）
├── home/       # 用户家目录，每个用户的个人文件（如 /home/username）
├── lib/        # 系统共享库文件，供程序运行时使用（如 libc.so, libm.so）
├── lost+found/ # 文件系统恢复目录，用于存放 fsck 恢复的文件
├── media/      # 可移动设备挂载点（如 USB, CD/DVD 挂载）
├── mnt/        # 临时挂载点，用于系统管理员挂载外部文件系统
├── opt/        # 第三方应用程序，通常为不依赖包管理器的安装软件
├── proc/       # 虚拟文件系统，包含内核和进程信息（如 /proc/cpuinfo, /proc/[pid]）
├── root/       # root 用户的家目录，系统管理员的个人文件
├── run/        # 运行时数据，存储系统当前会话的临时文件（如 PID 文件、锁文件）
├── sbin/       # 系统管理命令，供管理员（root 用户）使用（如 fsck, shutdown）
├── srv/        # 服务数据，存储由系统提供的服务的数据（如 /srv/ftp, /srv/http）
├── sys/        # 虚拟文件系统，提供系统硬件信息和内核参数
├── tmp/        # 临时文件，存储临时数据，系统重启时会清空
├── usr/        # 用户级程序和数据，包含大部分应用程序和共享文件
│   ├── bin/    # 用户命令二进制文件，用户程序（如 /usr/bin/ls, /usr/bin/gcc）
│   ├── lib/    # 用户程序的库文件（如 /usr/lib/libc.so）
│   ├── share/  # 共享文件，如文档、图标等（如 /usr/share/man）
├── var/        # 可变数据，存储日志、缓存、邮件队列等数据
│   ├── log/    # 日志文件（如 /var/log/syslog）
│   ├── spool/  # 打印队列或邮件队列（如 /var/spool/mail）
│   └── cache/  # 应用程序缓存（如 /var/cache/apt）
```

## 安装与设置

1. 安装ping等工具：`sudo apt-get install iputils-ping`
2. 安装网络工具（如ifconfig）：`sudo apt-get install net-tools`
3. 安装telnet：`apt-get install telnet`
4. `~/.bashrc`部分配置：

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

## 查看信息

1. 查看linux版本等信息：`cat /etc/os-release`
2. 查看当前使用的shell：`echo $SHELL`或者`echo $0`
3. 查看当前shell的PID：`echo $$`
4. linux查看当前文件夹中各文件大小：`du -sh ./*`或者用`ls -lh`
5. 查案磁盘使用情况：`df -h`
6. 查看内存：`free -h`
7. 查看系统系统区域设置（语言、字符编码、日期格式等）：`locale`
8. 查看系统支持语言：`locale -a`
9. 查看系统时间：`date`


## 文件操作

1. 文件解压
   1. zip文件解压：`unzip <filename>`
   2. tar文件解压：`tar -xvf <filename>`
   3. tar.gz文件解压：`tar -zxvf <filename>`
   4. rar文件解压：`unrar x <filename>`
   5. 7z文件解压：`7za x <filename>`
2. 文件复制
   1. 复制文件夹并排除特定子文件夹：`rsync -av --exclude="<subfolder1>" --exclude="<subfolder2>" --exclude="<subfolder3>" <source_dir> <destination_dir>`
   2. 例如：`rsync -av --exclude="*.T" --exclude="znr" --exclude=".git" ../up/bwy_v4/bwy ./`
   3. 其中，`-a`表示归档模式，表示递归复制并保持文件属性。`-v`表示详细模式，显示复制过程中的详细信息。
3. 文件传输（scp）
   1. 本地传到服务器：`scp -P <port> <filepath_windows> root@<remote_ip>:<filepath_linux>`
   2. 服务器传到本地：`scp -P <port> root@<remote_ip>:<filepath_linux> <filepath_windows>`
   3. 注意，因为开发机有防火墙之类的东西，所以没办法上传成功。要用这种方法传的话，需要在docker开个端口，可以直接用ssh连接docker，往docker映射到开发机的路径传就可以。所以上面的命令用的是root，而不是username
4. 文件传输（sz/rz，mobaxterm为例）
   1. 本地上传到服务器：`rz` && `ctrl + 鼠标右键` && `Send file using Z-modem` && `选择文件`
   2. 服务器下载到本地：`sz filename` && `ctrl + 鼠标右键` && `Receive file using Z-modem`
   3. 中途取消操作：`ctrl + x`按4到5次
5. 查看md5值：`md5sum <filename>`

## 网络端口操作

1. 查看本机ip：`ifconfig`
2. 查看所有端口使用情况：`netstat -tunlp`
   1. -t (tcp) 仅显示tcp相关选项
   2. -u (udp)仅显示udp相关选项
   3. -n 拒绝显示别名，能显示数字的全部转化为数字
   4. -l 仅列出在Listen(监听)的服务状态
   5. -p 显示建立相关链接的程序名
3. 查看某端口监听状态：`lsof -i -P -n | grep LISTEN`
   1. -i：显示与网络相关的文件
   2. -P：显示端口号而不是服务名（ssh就是22之类的）
   3. -n：不解析主机名（localhost就显示127.0.0.1之类的）
   4. 在端口占用的时候可以直接用`lsof -i:<port>`查看该端口情况
4. 查看防火墙规则：`iptables -L`
   1. 将输出的结果询问gpt是什么意思，给出以下回答
   2. 总的来说，输出显示了防火墙的配置情况，但并没有明确指出是否有针对公网 IP 地址的特定配置。要确保公网可以访问到指定端口，你需要确保在相应的链（比如 INPUT 链）中有允许公网 IP 地址访问指定端口的规则。
5. 测试能否连上某ip跟port：`telnet <ip> <port>`

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
wc <filename>
# 仅统计行数
wc -l <filename>
# 仅统计单词数
wc -w <filename>
# 仅统计字符数
wc -c <filename>
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
C:\                      # 系统盘（默认），存储 Windows 操作系统和应用程序
├── Program Files\        # 默认安装目录，存放 64 位应用程序（如 C:\Program Files\Google\Chrome）
├── Program Files (x86)\  # 存放 32 位应用程序（仅在 64 位 Windows 上存在）
├── Windows\              # 操作系统核心目录，包含 Windows 组件、配置和驱动
│   ├── System32\         # 64 位系统核心文件，存储 DLL、EXE、驱动程序（如 cmd.exe, notepad.exe）
│   ├── SysWOW64\         # 32 位系统核心文件，仅在 64 位 Windows 上存在
│   ├── WinSxS\           # Windows 组件存储，管理不同版本的 DLL 以防止冲突
│   ├── Temp\             # 系统临时文件，程序运行时的缓存文件
│   ├── Fonts\            # 存放系统字体文件（如 Arial.ttf, SimSun.ttc）
│   ├── Resources\        # 主题文件、界面资源（如壁纸、音效）
│   ├── Logs\             # 系统日志文件
│   ├── INF\              # 驱动安装信息文件
│   ├── Tasks\            # 计划任务存储目录
│   ├── SoftwareDistribution\ # Windows 更新文件缓存
│   └── Web\              # Edge 浏览器相关文件和壁纸存储
├── Users\                # 用户目录，存放用户的个人数据
│   ├── Administrator\    # 管理员账户目录
│   ├── Default\          # 默认用户配置模板，新建用户时会复制该目录
│   ├── Public\           # 共享文件夹，所有用户可访问
│   ├── [用户名]\         # 个人用户目录
│   │   ├── Desktop\      # 桌面文件
│   │   ├── Documents\    # 文档目录
│   │   ├── Downloads\    # 下载目录
│   │   ├── Pictures\     # 图片目录
│   │   ├── Videos\       # 视频目录
│   │   ├── Music\        # 音乐目录
│   │   ├── AppData\      # 应用程序数据（用户级）
│   │   │   ├── Local\    # 本地应用数据（如缓存文件）
│   │   │   ├── LocalLow\ # 低权限应用数据
│   │   │   └── Roaming\  # 可同步的用户数据（网络账户时可同步）
├── ProgramData\          # 公共应用数据（类似 Linux 的 `/var` 目录）
├── PerfLogs\             # 性能日志文件
├── Recovery\             # 系统恢复文件
├── System Volume Information\ # 备份和恢复点信息（普通用户无法访问）
├── Recycle Bin\          # 回收站，存放被删除但未永久清除的文件
└── Temp\                 # 系统级临时文件（可与 `C:\Windows\Temp` 不同）
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