- [tmux](#tmux)
  - [基本用法](#基本用法)
  - [基本命令](#基本命令)
  - [基本快捷键](#基本快捷键)
- [screen](#screen)


# tmux

[参考链接](https://www.ruanyifeng.com/blog/2019/10/tmux.html)

## 基本用法

**安装**

```bash
# 有 root 权限直接 apt install
apt install -y tmux
```

<!-- **从源码安装**

```shell
# 没有 root 权限

# 1. 创建文件夹
mkdir -p ~/.local/{bin,lib,include,src}

# 2. 编译依赖 libevent
cd ~/.local/src
wget https://github.com/libevent/libevent/releases/download/release-2.1.12-stable/libevent-2.1.12-stable.tar.gz
tar -xzf libevent-2.1.12-stable.tar.gz
cd libevent-2.1.12-stable
./configure --prefix=$HOME/.local --disable-openssl
make -j
make install

# 3. 编译依赖 ncurses
cd ~/.local/src
wget https://ftp.gnu.org/pub/gnu/ncurses/ncurses-6.4.tar.gz
tar -xzf ncurses-6.4.tar.gz
cd ncurses-6.4
./configure  --prefix=$HOME/.local  --with-shared  --with-termlib  --enable-pc-files --with-pkg-config-libdir=$HOME/.local/lib/pkgconfig
make -j
make install

# 编译依赖 m4
cd ~/.local/src
wget https://ftp.gnu.org/gnu/m4/m4-1.4.19.tar.gz
tar -xzf m4-1.4.19.tar.gz
cd m4-1.4.19
./configure --prefix=$HOME/.local
make -j
make install

# 4. 编译依赖 bison
cd ~/.local/src
wget https://ftp.gnu.org/gnu/bison/bison-3.8.2.tar.gz
tar -xzf bison-3.8.2.tar.gz
cd bison-3.8.2
export PATH=$HOME/.local/bin:$PATH
export M4=$HOME/.local/bin/m4
./configure --prefix=$HOME/.local
make -j
make install

# 5. 编译 tmux
cd ~/.local/src
wget https://github.com/tmux/tmux/releases/download/3.4/tmux-3.4.tar.gz
tar -xzf tmux-3.4.tar.gz
cd tmux-3.4
# 临时创建下编译的环境变量
export CPPFLAGS="-I$HOME/.local/include"
export LDFLAGS="-L$HOME/.local/lib"
export PKG_CONFIG_PATH="$HOME/.local/lib/pkgconfig"
export YACC=/mnt/afs_toolcall/zhengnairong/.local/bin/bison
export YFLAGS="-y"
# 配置并编译安装
./configure --prefix=$HOME/.local
make -j
make install
``` -->

**启动与退出**

```bash
tmux # 启动，底部有一个状态栏。状态栏的左侧是窗口信息（编号和名称），右侧是系统信息。
exit # 退出，或者Ctrl+d
```

**前缀键**

Tmux 窗口有大量的快捷键。所有快捷键都要通过前缀键唤起。默认的前缀键是`Ctrl+b`，即先按下`Ctrl+b`，快捷键才会生效。

举例来说，帮助命令的快捷用法是`Ctrl+b`然后再按`?`，会显示帮助信息。

本文的快捷键都写成：`Ctrl+b & <快捷键>`


**配置**

（不做ok的）

```bash
# 编辑文件: vim  ~/.tmux.conf

# 启用鼠标支持（选择文本、调整面板、切换窗口等）
set -g mouse on
# 自定义状态栏显示内容
set -g status-left "Session: #S | Windows: #W"
set -g status-right "Time: %H:%M:%S | Date: %Y-%m-%d"
# 设置xterm-keys
set -g xterm-keys on
# 保留会话，关闭时不会销毁
set -g destroy-unattached off

# 激活配置: tmux source-file ~/.tmux.conf
```


## 基本命令

```bash
# ======= 会话管理命令 ======= #
# 新建会话: tmux new -s <session_name>
# 分离会话: tmux detach # 会退出当前 Tmux 窗口，但是会话和里面的进程仍然在后台运行。（也可以用快捷键 Ctrl+b & d）
# 查看会话: tmux ls
# 接入会话: tmux attach -t <session_name or session_number>
# 杀死会话: tmux kill-session -t <session_name or session_number>
# 切换会话: tmux switch -t <session_name or session_number>
# 重命名会话: tmux rename-session -t <old_name> <new_name>

# ======= 窗口管理命令 ======= #
# 创建新窗口: tmux new-window -n <window_name>
# 切换到指定窗口: tmux select-window -t <window_name or window_number>
# 重命名窗口: tmux rename-window <new_name>

# ======= 窗格管理命令 ======= #
# 划分上下两个窗格: tmux split-window
# 划分左右两个窗格: tmux split-window -h
```


## 基本快捷键

```bash
# ======= 会话管理快捷键 ======= #
# 分离当前会话: Ctrl+b & d
# 列出所有会话: Ctrl+b & s
# 重命名当前会话: Ctrl+b & $

# ======= 窗口管理快捷键 ======= #
# 创建新窗口: Ctrl+b & c
# 切换到上一个窗口: Ctrl+b & p
# 切换到下一个窗口: Ctrl+b & n
# 切换到指定编号的窗口: Ctrl+b & <window_number>
# 从列表中选择窗口: Ctrl+b & w  # (这个好用！！！)
# 窗口重命名: Ctrl+b & ,

# ======= 窗格管理快捷键 ======= #
# 划分左右两个窗格: Ctrl+b & %
# 划分上下两个窗格: Ctrl+b & "
# 光标切换到其他窗格: Ctrl+b & ↑↓←→
# 关闭当前窗格: Ctrl+b & x
# 将当前窗格拆分为一个独立窗口: Ctrl+b & !
# 显示窗格编号: Ctrl+b & q
```

# screen

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