- [notes](#notes)
  - [Linux目录结构](#linux目录结构)
  - [Linux用户和组管理](#linux用户和组管理)
    - [用户管理](#用户管理)
    - [组管理](#组管理)
    - [权限管理](#权限管理)

# notes
[markdown数学公式和符号表](https://zhuanlan.zhihu.com/p/450465546)

1. ai_related：ai相关
2. dev_related：开发过程中的笔记
3. other：以前的一些笔记


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