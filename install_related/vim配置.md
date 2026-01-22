- [CMD](#cmd)
- [vim使用](#vim使用)
  - [vim模式选择](#vim模式选择)
  - [vim 的“公式思维”（关键）](#vim-的公式思维关键)
    - [常见操作符](#常见操作符)
    - [常见动作](#常见动作)
  - [vim基础操作](#vim基础操作)
    - [移动光标（很关键）](#移动光标很关键)
    - [删除、复制、粘贴](#删除复制粘贴)
    - [搜索与替换](#搜索与替换)
  - [vimdiff](#vimdiff)
- [vim 配置](#vim-配置)
  - [linux](#linux)
  - [windows的mobaXterm](#windows的mobaxterm)

# CMD

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


# vim使用

## vim模式选择

vim 有以下几种模式：

| 模式                | 作用                         | 进入方式                                                           | 退出方式 |
| ------------------- | ---------------------------- | ------------------------------------------------------------------ | -------- |
| 普通模式（Normal）  | 移动光标、删除、复制、粘贴   | 打开 vim 默认就是                                                  |
| 插入模式（Insert）  | 真正输入文字                 | i：在光标前插入 <br> a：在光标后插入 <br> o：新起一行并插入        | Esc      |
| 可视模式（Visual）  | 选中文本                     | v：字符级选择 <br> V：整行选择 <br> Ctrl + v：块选择（列编辑神器） | Esc      |
| 命令模式（Command） | 执行命令（保存、退出、搜索） | 普通模式下输入 `:`                                                 | Esc      |

## vim 的“公式思维”（关键）

**“公式”**

```shell
vim 命令 = 操作符 + 动作

d + w = 删除一个单词
y + $ = 复制到行尾
c + w = 删除一个单词并进入插入模式
```

### 常见操作符

| 操作符 | 含义              |
| ------ | ----------------- |
| d      | 删除              |
| y      | 复制              |
| c      | 修改（删 + 插入） |

### 常见动作

| 动作 | 含义             |
| ---- | ---------------- |
| w    | 单词             |
| $    | 行尾             |
| 0    | 行首             |
| i    | inside（如 ci"） |


## vim基础操作

### 移动光标（很关键）

**基本移动**

```shell
h  ←
l  →
j  ↓
k  ↑
```

**单词级移动**

```shell
w: 下一个单词开头（word）
b: 上一个单词开头（back / backward）
e: 当前单词结尾（end）
```

**行内移动**

```shell
0: 行首
^: 行首第一个非空字符
$: 行尾
```

**页面级移动**

```shell
gg: 文件开头（其实是一个操作符+命令）
G: 文件结尾
Ctrl + d: 向下翻半页（down）
Ctrl + u: 向上翻半页（up）
Ctrl + f: 向下翻一页（forward）
Ctrl + b: 向上翻一页（backward）
```

### 删除、复制、粘贴

**删除（d/delete）**

```shell
dd: 删除一整行
dw: 删除一个单词
d$: 删除到行尾
d0: 删除到行首
```

**复制（y/yank）**

```shell
yy: 复制一行
yw: 复制一个单词
y$: 复制到行尾
```

**粘贴（p/paste）**

```shell
P: 光标后粘贴（大写）
p: 光标前粘贴（小写）
```

**撤销 & 重做**

```shell
u: 撤销
Ctrl + r: 重做
```

### 搜索与替换

**搜索**

```shell
/word: 向下搜索
?word: 向上搜索
n: 下一个
N: 上一个
```

**替换（非常强）**

```shell
:%s/旧/新/g

%：全文
s：替换
g：全局
```


## vimdiff

vimdiff的一些设置：

```shell
# 命令行模式下

## 有时候没配置的话高亮太难看，可以改主题
:colorscheme desert
:colorscheme evening
:colorscheme elflord
:colorscheme industry

## 相同内容折叠与否（需要两边都设置一下）
:set nofoldenable   # 相同内容不折叠
:set foldenable     # 相同内容折叠

## 是否换行显示
:set wrap           # 换行显示
:set nowrap         # 关闭换行显示
```

# vim 配置

可以直接参考 repo：[https://github.com/amix/vimrc](https://github.com/amix/vimrc)

下面给一些我之前用的配置（需要插件），没必要其实。

## linux

1. 创建文件`~/.vimrc`（里面有些地方可以改的，看看注释）

```bash
set nocompatible

filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin('~/.vim/bundle')
Plugin 'VundleVim/Vundle.vim'
Plugin 'vim-airline/vim-airline'
Plugin 'vim-airline/vim-airline-themes'
Plugin 'Lokaltog/vim-easymotion'
Plugin 'The-NERD-tree'
Plugin 'Yggdroot/indentLine'
Plugin 'jiangmiao/auto-pairs'
Plugin 'scrooloose/nerdcommenter'
Plugin 'Valloric/YouCompleteMe'
Plugin 'morhetz/gruvbox'
call vundle#end()
filetype plugin indent on

set history=50                  " How many lines of history to remember
set confirm                     " Ask for confirmation in some situations (:q)
set ignorecase smartcase        " case insensitive search, except when mixing cases
set modeline                    " we allow modelines in textfiles to set vim settings
set hidden                      " allows to change buffer without saving
set mouse=a                     " enable mouse in all modes
set noerrorbells                " don't make noise
set novisualbell                " don't blink
set t_Co=256                    " Enable 256 color mode
set hlsearch
set cursorline
set cursorcolumn
set nowrap
syntax on

" Global indent settings
set tabstop=4 softtabstop=4 shiftwidth=4 expandtab

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" File settings
" """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
set encoding=utf-8              " Let Vim use utf-8 internally
set fileencoding=utf-8          " Default for new files
set termencoding=utf-8          " Terminal encoding
set fileformats=unix,dos,mac    " support all three, in this order
set fileformat=unix             " default file format

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" UI/Colors
" """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
set wildmenu                    " Show suggestions on TAB for some commands
set ruler                       " Always show current positions along the
set cmdheight=1                 " the command bar is 1 high
set number                      " turn on line numbers
"set nonumber                    " turn off line numbers (problematic with
set lazyredraw                  " do not redraw while running macros (much
set backspace=indent,eol,start  " make backspace work normal
set whichwrap+=<,>,h,l          " make cursor keys and h,l wrap over line
set report=0                    " always report how many lines where changed
set fillchars=vert:\ ,stl:\ ,stlnc:\    " make the splitters between windows
set laststatus=2                " always show the status line
set scrolloff=10                " Start scrolling this number of lines from
colorscheme gruvbox
set background=dark    " Setting dark mode
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Plugins
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"缩进指示线"
let g:indentLine_char='┆'
let g:indentLine_enabled = 1

map <F4> <leader>ci <CR>

let mapleader=','
" vim-airline/vim-airline
"let g:airline_section_b = '%b 0x%B'
"let g:airline_section_b = airline#section#create_right(['fileencoding','fileformat'])
"let g:airline_section_b = airline#section#create_right(['tagbar','gutentags', 'grepper', 'filetype'])
let g:airline_section_x = '%b 0x%B'
let g:airline#extensions#tabline#enabled = 1        " Enhanced top tabline
let g:airline#extensions#tabline#buffer_nr_show = 1 " Show buffer number intabline
let g:airline_powerline_fonts = 1                   " Use powerline fonts(requires local font)

"""""YCM plugin"""""""
"默认配置文件路径
let g:ycm_global_ycm_extra_conf = '~/.vim/ycm_extra_conf.py' "注意这里需要自己配置.ycm_extra_conf.py文件路径，文件见附件
 
"打开vim时不再询问是否加载ycm_extra_conf.py配置
let g:ycm_confirm_extra_conf=0
set completeopt=longest,menu
"python解释器路径"
let g:ycm_path_to_python_interpreter='/opt/anaconda3/bin/python' "注意这里需要配置自己常用的python路径，这样ycm可以查找到你是用的Python包，方便coding的时候自动补全
let g:ycm_python_binary_path='/opt/anaconda3/bin/python'
"是否开启语义补全"
let g:ycm_seed_identifiers_with_syntax=1
"是否在注释中也开启补全"
let g:ycm_complete_in_comments=1
let g:ycm_collect_identifiers_from_comments_and_strings = 0
"开始补全的字符数"
let g:ycm_min_num_of_chars_for_completion=2
"补全后自动关机预览窗口"
let g:ycm_autoclose_preview_window_after_completion=1
" 禁止缓存匹配项,每次都重新生成匹配项"
let g:ycm_cache_omnifunc=0
"字符串中也开启补全"
let g:ycm_complete_in_strings = 1
"离开插入模式后自动关闭预览窗口"
autocmd InsertLeave * if pumvisible() == 0|pclose|endif
"回车即选中当前项"
inoremap <expr> <CR> pumvisible() ? '<C-y>' : '<CR>'
"上下左右键行为"
inoremap <expr> <Down> pumvisible() ? '\<C-n>' : '\<Down>'
inoremap <expr> <Up> pumvisible() ? '\<C-p>' : '\<Up>'
inoremap <expr> <PageDown> pumvisible() ? '\<PageDown>\<C-p>\<C-n>' : '\<PageDown>'
inoremap <expr> <PageUp> pumvisible() ? '\<PageUp>\<C-p>\<C-n>' : '\<PageUp>'

```

1. 创建文件`~/.vim/ycm_extra_conf.py`

```bash
# This file is NOT licensed under the GPLv3, which is the license for the rest
# of YouCompleteMe.
#
# Here's the license text for this file:
#
# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <http://unlicense.org/>

import os.path as p
import subprocess

DIR_OF_THIS_SCRIPT = p.abspath( p.dirname( __file__ ) )
DIR_OF_THIRD_PARTY = p.join( DIR_OF_THIS_SCRIPT, 'third_party' )

def GetStandardLibraryIndexInSysPath( sys_path ):
  for index, path in enumerate( sys_path ):
    if p.isfile( p.join( path, 'os.py' ) ):
      return index
  raise RuntimeError( 'Could not find standard library path in Python path.' )

def PythonSysPath( **kwargs ):
  sys_path = kwargs[ 'sys_path' ]

  dependencies = [ p.join( DIR_OF_THIS_SCRIPT, 'python' ),
                   p.join( DIR_OF_THIRD_PARTY, 'requests-futures' ),
                   p.join( DIR_OF_THIRD_PARTY, 'ycmd' ),
                   p.join( DIR_OF_THIRD_PARTY, 'requests_deps', 'idna' ),
                   p.join( DIR_OF_THIRD_PARTY, 'requests_deps', 'chardet' ),
                   p.join( DIR_OF_THIRD_PARTY,
                           'requests_deps',
                           'urllib3',
                           'src' ),
                   p.join( DIR_OF_THIRD_PARTY, 'requests_deps', 'certifi' ),
                   p.join( DIR_OF_THIRD_PARTY, 'requests_deps', 'requests' ) ]

  # The concurrent.futures module is part of the standard library on Python 3.
  interpreter_path = kwargs[ 'interpreter_path' ]
  major_version = int( subprocess.check_output( [
    interpreter_path, '-c', 'import sys; print( sys.version_info[ 0 ] )' ]
  ).rstrip().decode( 'utf8' ) )
  if major_version == 2:
    dependencies.append( p.join( DIR_OF_THIRD_PARTY, 'pythonfutures' ) )

  sys_path[ 0:0 ] = dependencies
  sys_path.insert( GetStandardLibraryIndexInSysPath( sys_path ) + 1,
                   p.join( DIR_OF_THIRD_PARTY, 'python-future', 'src' ) )

  return sys_path
```

1. 安装插件：`git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim`
2. 终端输入`vim`，进入命令模式后输入`:PluginInstall`
3. 等待插件安装完成即可。
4. 有时候会出现vim版本低用不了`YouCompleteMe`的问题，可以[升级vim](https://zhuanlan.zhihu.com/p/550341865)：
   1. `sudo apt-get install software-properties-common`
   2. `sudo add-apt-repository ppa:jonathonf/vim`
   3. `sudo apt update`
   4. `sudo apt install vim`
5. 有时候还会出现`YouCompleteMe`的`server`有问题，是因为没有编译：
   1. 安装cmake：`sudo apt install cmake`
   2. 安装C++编译器：`sudo apt-get install g++`
   3. 切换到YCM路径：`cd ~/.vim/bundle/YouCompleteMe`
   4. 安装：`python install.py [--force-sudo]`

## windows的mobaXterm

1. 流程是一样的，但是在mobaXterm操作的时候，会出现错误`E492: 不是编辑器的命令: ^M`，这是因为linux和windows文件换行的区别，需要将这些文件从dos装成unix，采用以下命令（有些系统的dos2unix工具不一样，看看下面哪个能用就用哪个）：
   1. `dos2unix -o -r /path/to/your/folder`
   2. `find ~/.vim/bundle -type f -exec dos2unix {} \;`
2. 在YCM编译的时候需要通过装Visual Studio来安装C++相关的东西，相比于linux会麻烦点。（编译完好像也不一定有用就是了，windows没搞明白）
3. 最简单的就是不要用YCM，大不了不要自动补全，把`~/.vimrc`最上面的`Plugin 'Valloric/YouCompleteMe’`注释掉，下面相关有影响的也注释掉。
4. 或者用coc（未完成）：
   1. 安装nodejs：`curl -sL install-node.now.sh/lts | bash`
   2. 安装yarn：`curl --compressed -o- -L https://yarnpkg.com/install.sh | bash`
   3. 安装另一个插件管理器：`curl -fLo ~/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim`（不要改，就这样）
