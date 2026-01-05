- [安装与配置](#安装与配置)
- [一些命令](#一些命令)
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
# 可以加源来保证 gvm install 速度：export GVM_GO_GETTER="wget -O - https://mirrors.tuna.tsinghua.edu.cn/golang/go\$VERSION.src.tar.gz"
gvm install                 # 列出所有可用go版本（包括旧版本）
gvm list                    # 显示已安装的go版本
gvm install go1.22.1        # 安装指定的go版本
gvm use go1.22.1            # 使用指定的go版本（当前shell有效）
gvm use go1.22.1 --default  # 设置默认go版本（永久生效）
gvm uninstall go1.22.1      # 卸载指定go版本
gvm pkgset list             # 显示当前版本下的包集
```

**手动安装go (1.22.1版本为例)**

方法一（麻烦不推荐）：

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

方法二（推荐）：

```bash
# 手动下载go
cd /tmp
wget https://golang.google.cn/dl/go1.22.1.linux-amd64.tar.gz
mv go1.22.1.linux-amd64.tar.gz ~./gvm/gos/go1.22.1.linux-amd64.tar.gz
# 手动下载了包之后 gvm install 就会优先识别这里的包
gvm install go1.22.1
# 有时候会编译失败，可以慢慢升级版本
# 因为高版本的编译都是依赖低版本的，一下子装太高级的会编译不过
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
# 不过尽量不要写死 goroot
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

# 一些命令

**gvm相关命令**（上面有）

**go相关命令**

```bash
# 查看go环境
go env                                             # 查看所有环境变量
go env GOPATH                                      # 查看GOPATH
go env GOPROXY                                     # 查看代理设置

# 查看go版本
go version

# 编译和运行
go run <file.go>                                   # 直接运行（不生成可执行文件）
go build                                           # 编译当前目录（生成可执行文件）
go build -o <output_name>                          # 指定输出文件名
go build <package_path>                            # 编译指定包
go install                                         # 编译并安装到$GOPATH/bin

# 跨平台编译
GOOS=linux GOARCH=amd64 go build -o <output>       # 编译Linux版本
GOOS=windows GOARCH=amd64 go build -o <output>     # 编译Windows版本
GOOS=darwin GOARCH=amd64 go build -o <output>      # 编译macOS版本
    # GOOS选项：linux, windows, darwin, freebsd等
    # GOARCH选项：amd64, 386, arm, arm64等

# 代码格式化
go fmt <file.go>                                   # 格式化指定文件
go fmt ./...                                       # 格式化当前目录及子目录所有.go文件
gofmt -w <file.go>                                 # 格式化并写入文件

# 代码检查
go vet <file.go>                                   # 检查代码问题
go vet ./...                                       # 检查当前目录及子目录

# 测试相关
go test                                            # 运行当前目录测试
go test ./...                                      # 运行所有测试
go test -v                                         # 详细输出
go test -run <TestName>                            # 运行指定测试
go test -cover                                     # 显示测试覆盖率
go test -coverprofile=coverage.out                 # 生成覆盖率文件
go tool cover -html=coverage.out                   # 查看覆盖率HTML报告
go test -bench=.                                   # 运行性能测试
go test -cpuprofile=cpu.prof                       # CPU性能分析
go test -memprofile=mem.prof                       # 内存性能分析

# 获取包信息
go list                                            # 列出当前包
go list ./...                                      # 列出所有包
go list -m all                                     # 列出所有依赖模块
go list -m -versions <module_path>                 # 列出模块的所有版本

# 清理缓存
go clean                                           # 清理编译文件
go clean -cache                                    # 清理构建缓存
go clean -modcache                                 # 清理模块缓存
go clean -testcache                                # 清理测试缓存

# 文档相关
go doc <package>                                   # 查看包文档
go doc <package>.<function>                        # 查看函数文档
godoc -http=:6060                                  # 启动本地文档服务器（需安装godoc）
```

**go mod模块管理**

```bash
# 初始化模块
go mod init <module_name>                          # 创建go.mod文件
go mod init github.com/username/project            # 常用格式

# 管理依赖
go mod tidy                                        # 添加缺失的模块，删除无用的模块（推荐常用）
go mod download                                    # 下载依赖到本地缓存
go mod vendor                                      # 将依赖复制到vendor目录
go mod verify                                      # 验证依赖是否被修改

# 查看依赖
go mod graph                                       # 打印模块依赖图
go list -m all                                     # 列出所有依赖模块
go list -m -json all                               # JSON格式输出

# 更新依赖
go get <package>@latest                            # 更新到最新版本
go get <package>@<version>                         # 更新到指定版本
go get <package>@<commit_hash>                     # 更新到指定commit
go get -u                                          # 更新所有直接依赖
go get -u ./...                                    # 更新所有依赖（包括间接依赖）

# 替换模块（本地开发常用）
go mod edit -replace=<old_module>=<new_module>     # 替换模块路径
go mod edit -replace=example.com/m=../m           # 使用本地路径
go mod edit -dropreplace=<module>                  # 删除替换规则

# go.mod 示例
# module github.com/username/myproject
# 
# go 1.21
# 
# require (
#     github.com/gin-gonic/gin v1.9.1
#     github.com/go-sql-driver/mysql v1.7.1
# )
# 
# replace github.com/some/module => ../local/module
```

**go get包管理（模块模式）**

```bash
# 安装包
go get <package_path>                              # 安装包到当前模块
go get <package_path>@latest                       # 安装最新版本
go get <package_path>@v1.2.3                       # 安装指定版本
go get -u <package_path>                           # 更新到最新小版本
go get -u=patch <package_path>                     # 仅更新补丁版本

# 安装工具
go install <package_path>@latest                   # 安装工具到$GOPATH/bin
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# 常用工具安装
go install golang.org/x/tools/gopls@latest         # LSP服务器
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest  # 代码检查工具
go install github.com/swaggo/swag/cmd/swag@latest  # API文档生成
```

**常用第三方工具**

```bash
# golangci-lint - 代码质量检查（推荐）
golangci-lint run                                  # 运行所有检查器
golangci-lint run --disable-all --enable=errcheck  # 只运行指定检查器

# air - 热重载工具（开发时自动编译）
air                                                # 需要先安装：go install github.com/cosmtrek/air@latest

# swag - 生成Swagger API文档
swag init                                          # 生成docs目录
```


# 项目开发

具体查看 [simple_go](https://github.com/NairongZheng/learning/tree/main/go_related/simple_go)
