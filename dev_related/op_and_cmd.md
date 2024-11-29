[toc]
# linux操作相关

## 安装与设置

1. 安装ping等工具：`sudo apt-get install iputils-ping`
2. 安装网络工具（如ifconfig）：`sudo apt-get install net-tools`
3. 安装telnet：`apt-get install telnet`
4. 设置终端提示符样式（`~/.bashrc`里）：`PS1="\e[1;37m[\e[m\e[1;32m\u\e[m\e[1;33m@\e[m\e[1;35m\h\e[m:\e[m\$PWD\e[m\e[1;37m]\e[m\e[1;36m\e[m$ "`
5. 禁止分页显示（`~/.bashrc`里，失效的时候要重新`source ~/.bashrc`一下）：
   1. `export PAGER=cat`
   2. `export MANPAGER=cat`
   3. `export GIT_PAGER=cat`

## 查看信息

1. 查看linux版本等信息：`cat /etc/os-release`
2. 查看当前使用的shell：`echo $SHELL`或者`echo $0`
3. 查看当前shell的PID：`echo $$`
4. linux查看当前文件夹中各文件大小：`du -sh ./*`或者用`ls -lh`
5. 查案磁盘使用情况：`df -h`
6. 查看内存：`free -h`

## 用户管理

1. 新建用户：`sudo useradd -m -s /bin/bash <user_name>` && `sudo mkdir -p /home/<user_name>`
2. 删除用户（保留主目录）：`sudo userdel <user_name>`
3. 删除用户及其主目录：`sudo userdel -r <user_name>`
4. 设置用户进root无需密码：
   1. `sudo visudo`
   2. `<%sudo ALL=(ALL:ALL) ALL>`下面添加`<your_username> ALL=(ALL) NOPASSWD:ALL`
   3. `ctrl + x` 保存再回车就退出
   4. 或者直接用命令：`sudo echo "<user_name> ALL=(ALL:ALL) NOPASSWD: ALL" >>/etc/sudoers`

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

## ssh操作

1. 重启ssh服务：`/etc/init.d/ssh restart`（`/etc/ssh/sshd_config`）
2. 生成密钥对：`ssh-keygen -t rsa [-f </path/to/private_key> | -C <some tag such as email>]`
3. 连接命令：
   1. 密码登录（可选为指定算法）：`ssh <user_name>@<remote_ip> -p <remote_port> [-o HostKeyAlgorithms=+ssh-rsa]` 
   2. 密钥登录（直接登跳板机啥也没有）：`ssh <user_name>@<remote_ip> -p <remote_port> -i <private_key_path>`
   3. 密钥通过跳板机登录开发机：`ssh <user_name>@<dev_ip> -i <private_key_path> -o ProxyCommand="ssh <user_name>@<jumpserver_ip> -p <jumpserver_port> -i <private_key_path> -q -W <dev_ip>:<dev_port>"`

## vim操作

1. 分屏打开另一个文件
   1. 垂直分屏：`:vsplit <filename> | :vsp <filename>`
   2. 水平分屏：`:split <filename> | :sp <filename>`
2. 切换分屏
   1. `Ctrl-w w` 或者 `Ctrl-w Ctrl-w`：在不同的分屏之间循环切换。
   2. `Ctrl-w h`：切换到左边的分屏。
   3. `Ctrl-w j`：切换到下面的分屏。
   4. `Ctrl-w k`：切换到上面的分屏。
   5. `Ctrl-w l`：切换到右边的分屏。
3. 调整分屏大小
   1. `Ctrl-w +`：增加当前分屏的高度。
   2. `Ctrl-w -`：减少当前分屏的高度。
   3. `Ctrl-w >`：增加当前分屏的宽度。
   4. `Ctrl-w <`：减少当前分屏的宽度。

## 其他操作

1. [重定向输出到黑洞](https://blog.csdn.net/longgeaisisi/article/details/90519690)：`/dev/null 2>&1`

# windows操作相关

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