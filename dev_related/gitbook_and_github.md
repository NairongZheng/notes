[toc]
参考链接：[https://einverne.github.io/gitbook-tutorial/](https://einverne.github.io/gitbook-tutorial/)

# 安装

## 安装npm与nodejs

参考[链接](https://blog.csdn.net/HuangsTing/article/details/113857145)安装nodejs等，这个好用。可以进行版本控制。

[nvm官方](https://github.com/coreybutler/nvm-windows/releases)下载，建议直接默认路径，不然好像环境路径会有问题。

```bash
# 安装完后查看版本号
nvm version
# 查看可以安装版本号
nvm list available
# 选择需要的版本进行安装
nvm install 10.21.0
# 查看已经安装了哪些版本（可以切换的）
nvm lsit
# 选择安装好的版本使用
nvm use 10.21.0
# 查看nodejs版本
node -v
# 查看npm版本
npm -v
# 都有的话说明安装成功
```

用上面方法安装的npm跟nodejs可以进行版本控制，在安装gitbook的时候版本不行还可以切换，使用别的需要nodejs的应用也可以用这个，所以比较方便。

## 安装gitbook

```bash
# 安装gitbook
npm install gitbook-cli -g
# 查看安装的gitbook的版本（就是这一步安装gitbook很容易出错）
gitbook -V
# 安装成功显示如下
# C:\Users\Administrator>gitbook -V
# CLI version: 2.3.2
# GitBook version: 3.2.3
```

我在安装gitbook的时候就出现了上面提到的版本问题，可以参考[链接1](https://blog.csdn.net/weixin%5C_42349568/article/details/108414441)跟[链接2](https://blog.csdn.net/Lowerce/article/details/107579261)。

# github与gitbook设置

参考链接：[https://www.bilibili.com/video/BV1xk4y1i7Lo/](https://www.bilibili.com/video/BV1xk4y1i7Lo/)

## github初始化仓库

新建一个仓库并拉到本地。可以先写一些配置文件，如`.gitignore`，`.gitbook.yaml`等

```bash
# .gitbook.yaml
root: ./docs/
# 这样gitbook就会创建一个docs目录存放md文件，根目录就不会太杂
```

然后将其推送到远端仓库

## gitbook设置

1. 选择左边菜单栏的`integrations`
2. 安装GitHub Sync
3. `configuration`里选择要同步的book
4. 跳转GitHub授权并安装
5. 选择GitHub仓库用来与该book同步，分支选择主分支就可以
6. 第一次同步一定要选择从gitbook同步到GitHub

## 本地编译gitbook

经过上面的配置，每次在gitbook网站上进行修改，都会同步到GitHub。

所以每次改完之后，本地仓库都要pull一下，进行更新。

第一次更新会发现，多了一个`docs`文件夹，就是上面配置。

我习惯主分支用来保存这些干净的文件夹，然后用一个别的分支来进行编译，保存编译的那些文件。

并且后面要用GitHub的pages进行托管也一定要用`gh-pages`分支，所以可以直接`checkout`到这个新分支。

```bash
# 切换分支
$ git checkout [-b] gh-pages
#
$ gitbook serve
Live reload server started on port: 35729
Press CTRL+C to quit ...

info: 7 plugins are installed
info: loading plugin "livereload"... OK
info: loading plugin "highlight"... OK
info: loading plugin "search"... OK
info: loading plugin "lunr"... OK
info: loading plugin "sharing"... OK
info: loading plugin "fontsettings"... OK
info: loading plugin "theme-default"... OK
info: found 5 pages
info: found 2 asset files
info: >> generation finished with success in 0.6s !

Starting server ...
Serving book on http://localhost:4000
```

就可以在`http://localhost:4000`查看了

## 发布到github pages

一样在`docs`文件夹内执行命令：

```bash
$ gitbook build
info: 7 plugins are installed
info: 6 explicitly listed
info: loading plugin "highlight"... OK
info: loading plugin "search"... OK
info: loading plugin "lunr"... OK
info: loading plugin "sharing"... OK
info: loading plugin "fontsettings"... OK
info: loading plugin "theme-default"... OK
info: found 5 pages
info: found 2 asset files
info: >> generation finished with success in 0.6s !
```

编译后会自动生成一个`_book`文件夹，将文件夹内所有东西都复制到`gh-pages分支`根目录，然后push该分支。

> 之后每次gitbook网站上修改，会自动同步到GitHub该仓库的主分支。
>
>
> 每次要更新pages都需要在本地先拉最新的主分支
>
> checkout到gh-pages分支
>
> merge主分支
>
> 再gitbook build编译上传