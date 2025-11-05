- [General](#general)
- [Windows](#windows)
  - [MobaXterm](#mobaxterm)
    - [tmux](#tmux)
- [Mac](#mac)
  - [设置](#设置)
  - [shell](#shell)
  - [Miniconda](#miniconda)
  - [iterm2](#iterm2)
    - [tmux](#tmux-1)
  - [docker](#docker)
- [Other](#other)
  - [Vscode/cursor](#vscodecursor)
    - [cursor的一些设置](#cursor的一些设置)
    - [在cursor手动同步安装vscode的扩展](#在cursor手动同步安装vscode的扩展)

# General

1. sinpaste：[https://zh.snipaste.com/](https://zh.snipaste.com/)
2. 飞书：[https://www.feishu.cn/download](https://www.feishu.cn/download)
3. vscode：[https://code.visualstudio.com/](https://code.visualstudio.com/)
4. cursor：[https://cursor.com/cn](https://cursor.com/cn)
5. todesk：[https://www.todesk.com/](https://www.todesk.com/)

# Windows

1. winrar：[https://www.rarlab.com/download.htm](https://www.rarlab.com/download.htm)
2. mobaxterm：[https://mobaxterm.mobatek.net/](https://mobaxterm.mobatek.net/)
3. notepad++：[https://notepad-plus-plus.org/downloads/](https://notepad-plus-plus.org/downloads/)

## MobaXterm

### tmux

mobaxterm非常好地支持了鼠标左键选中复制，右键粘贴（需要勾选一下

但是在使用tmux的时候，鼠标的功能会遇到两难：
1. 如果需要用mobaxterm一模一样的”左键复制，右键粘贴“，那么就用不了tmux的鼠标，选中窗口切换等功能，就需要使用快捷键比较麻烦
2. 如果需要使用鼠标滚轮以及tmux直接鼠标切换窗口，那么就没办法”左键复制，右键粘贴“

那么根据我的习惯，最终采用比较能接受的方式，也就是上面的2

`~/.tmux.conf`配置如下：

```shell
# 编辑文件: vim  ~/.tmux.conf

set -g default-terminal "xterm-256color"

# 让tmux可以捕获鼠标活动
set -g mouse on
set -ga terminal-overrides ',xterm*:smcup@:rmcup@'

# 自定义状态栏显示内容
set -g status-left "Session: #S | Windows: #W"
set -g status-right "Time: %H:%M:%S | Date: %Y-%m-%d"

# 保留会话，关闭时不会销毁
set -g destroy-unattached off

# 激活配置: tmux source-file ~/.tmux.conf
```

这样的话就需要使用shift来配合鼠标的左右键了

# Mac

1. Mos：[https://mos.caldis.me/](https://mos.caldis.me/)（支持触控板跟鼠标都自然滚动）
2. AppCleaner：[https://freemacsoft.net/appcleaner/](https://freemacsoft.net/appcleaner/)（卸载清理软件）
3. iterm2：[https://iterm2.com/](https://iterm2.com/)
4. notepad--：[https://github.com/cxasm/notepad--](https://github.com/cxasm/notepad--)

## 设置
- 设置-触控板-光标与点按：轻点来点按
- 设置-鼠标：自然滚动
- 设置-辅助功能-指针控制-触控板选项：使用触控板进行拖移-三指拖移
- 访达-设置-边栏

## shell

安装brew

```shell
# 安装brew（中间是需要输入mac的密码的，安装也是需要等一会儿的
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# 然后会提示“Next steps”，按照执行即可，我这里是
echo >> /Users/zhengnairong/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/zhengnairong/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

安装常用命令行工具

```shell
brew install wget        # wget
brew install tmux        # tmux
brew install bash        # bash # 自带的/bin/bash版本低，可以使用这个命令安装新的bash，然后在～/.zshrc里面添加 export PATH="/opt/homebrew/bin:$PATH" 来使用新安装的bash
```

## Miniconda

```shell
cd ~
# 可以自己去网站挑选版本，命令行使用 uname -m 查看是arm64还是x86_64
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py312_24.9.2-0-MacOSX-arm64.sh
bash Miniconda3-py312_24.9.2-0-MacOSX-arm64.sh -b -p $HOME/miniconda3
    # -b：静默安装，不需要交互
    # -p：指定安装目录

# 有时候不需要添加，直接source即可
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# conda可以添加envs的路径来使用别的环境
conda config --add envs_dirs ${path/to/conda/envs}
```

## iterm2

1. 设置option+箭头移动一个字词
   1. iTerm2 → Settings → Profiles → Keys → Key Bindings
   2. 添加新的 Key Mapping（左下角的加号）
      1. Keyboard Shortcut：Option+Left
      2. Action：Send Escape Sequence
      3. Esc+Sequence：输入 b
   3. 同理 Option+Right → Esc+Sequence → f
   4. 但是这种方式在vim下面会跟vim有冲突（主要是向右），就需要配置~/.vimrc

```shell
" 设置超时时间
set timeout
set timeoutlen=200

" 插入模式下按词移动
inoremap <Esc>b <C-o>b
inoremap <Esc>f <C-o>w

" 普通模式下也可以
nnoremap <Esc>b b
nnoremap <Esc>f w
```

2. 设置左键复制到剪切板，右键从剪切板粘贴
   1. 左键复制到剪切板上，默认功能，具体在：Settings-General-Selection里面
   2. 右键粘贴到剪切板：Settings-Pointer-Bindings，左下角点击添加，然后自己选择
3. 在选项卡之间进行切换：command+箭头

### tmux

由于iterm2比较灵活，这么设置完了之后，在tmux里面也是可以使用这些功能的。

我的tmux配置：

```shell
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
```

## docker

```shell
# 安装
brew install --cask docker # --cask: 用于安装带有图形界面的桌面应用程序。macos需要使用这个来启动docker的守护进程
# 启动
open /Applications/Docker.app # 或者在电脑双击打开也行
```

# Other

## Vscode/cursor

### cursor的一些设置

1. 侧边栏：settings-workbench-apperance-activity bar
2. 设置中文：安装中文包之后在设置中搜索configure display language

### 在cursor手动同步安装vscode的扩展

1. 在环境变量中安装code或者cursor命令：ctrl+shift+p，shell commond: install ....
2. 导出vscode扩展：code --list-extensions > vscode-extensions.txt
3. 在cursor下载vscode一样的扩展：
   1. Linux/macos：cat vscode-extensions.txt | xargs -L 1 cursor --install-extension
   2. windows的cmd：for /f "tokens=*" %i in (vscode-extensions.txt) do cursor --install-extension %i
