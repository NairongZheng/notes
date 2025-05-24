- [安装与配置](#安装与配置)
- [项目开发](#项目开发)


# 安装与配置

**安装gvm**

gvm (Go Version Manager) 安装：

```bash
# 安装相关依赖
apt install curl git make binutils bison gcc build-essential
# 安装gvm
cd /tmp
bash < <(curl -s -S https://raw.githubusercontent.com/moovweb/gvm/master/binscripts/gvm-installer)
# curl -s -S https://raubusercontent.com/moovweb/gvm/master/binscripts/gvm-installer > gvm-installer.sh
# bash gvm-installer.sh
# 设置环境变量
source ~/.gvm/scripts/gvm
# 设置完之后会在~/.bashrc中自动添加类似如下内容：
# [[ -s "$HOME/.gvm/scripts/gvm" ]] && source "$HOME/.gvm/scripts/gvm"
```

**gvm相关命令**

```bash
gvm install                 # 列出所有可用go版本（包括旧版本）
gvm list                    # 显示已安装的go版本
gvm install go1.22.1        # 安装指定的go版本
gvm use go1.22.1            # 使用指定的go版本（当前shell有效）
gvm use go1.22.1 --default  # 设置默认go版本（永久生效）
gvm uninstall go1.22.1      # 卸载指定go版本
gvm pkgset list             # 显示当前版本下的包集
```

**手动安装go (1.22.1版本为例)**

```bash
# 手动下载go
cd /tmp
wget https://golang.google.cn/dl/go1.22.1.linux-amd64.tar.gz
mkdir -p ~/.gvm/gos/go1.22.1
tar -C ~/.gvm/gos/go1.22.1 -xzf go1.22.1.linux-amd64.tar.gz --strip-components=1
    # --strip-components=1：去掉压缩包中的顶层 go/ 文件夹，使内容直接放在 go1.22.1/ 下。
# 添加到gvm管理
cd ~/.gvm/environments
vim go1.22.1
# 添加以下内容
    # # Automatically generated file. DO NOT EDIT!
    # export GVM_ROOT; GVM_ROOT="/home/damonzheng/.gvm"
    # export gvm_go_name; gvm_go_name="go1.22.1"
    # export gvm_pkgset_name; gvm_pkgset_name="global"
    # export GOROOT; GOROOT="/home/damonzheng/.gvm/gos/go1.22.1"
    # export GOPATH; GOPATH="/home/damonzheng/.gvm/pkgsets/go1.22.1/global"
    # export PATH; PATH="/home/damonzheng/.gvm/pkgsets/go1.22.1/global/bin:$GOROOT/bin:/home/damonzheng/.gvm/bin:$PATH"
# 使配置生效
source ~/.gvm/environments/go1.22.1
# 设置该版本go为默认版本
gvm use go1.22.1 --default
```

这样就可以正常使用gvm来管理这个版本的go了。

**设置go模块下载代理**

linux终端永久有效：

```bash
vim ~/.bashrc
# 添加以下内容
export GOPROXY=https://goproxy.cn,direct
# 激活设置
source ~/.bashrc
```

windows终端临时有效：

```bash
set GOPROXY=https://goproxy.cn,direct
# 验证设置
go env GOPROXY
```

远程vscode有效：

```bash
# 路径：/home/damonzheng/.vscode-server/data/Machine/settings.json
# 打开方式：左下角齿轮->settings->搜索 Go: Tools Env Vars
{
    "go.goroot": "/home/damonzheng/.gvm/gos/go1.22.1",
    "go.gopath": "/home/damonzheng/.gvm/pkgsets/go1.22.1/global",
    "go.toolsEnvVars": {
        "GOPROXY": "https://goproxy.cn,direct"
    }
}
```

**vscode禁用go包跳转网页**

```bash
# Ctrl+, 打开vscode设置项，搜索gopls，添加以下项
"gopls": {
    "ui.navigation.importShortcut": "Definition"
}
```

**vscode配置go高亮**

```bash
go install golang.org/x/tools/gopls@latest
# Ctrl+, 打开vscode设置项，搜索gopls，添加以下项
"gopls": {
    "ui.navigation.importShortcut": "Definition"
}
```

# 项目开发

具体查看 [simple_go](https://github.com/NairongZheng/learning/tree/main/go_related/simple_go)
