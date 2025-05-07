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

终端有效：

```bash
vim ~/.bashrc
# 添加以下内容
export GOPROXY=https://goproxy.cn,direct
# 激活设置
source ~/.bashrc
```

远程vscode有效：

```json
// 路径：/home/damonzheng/.vscode-server/data/Machine/settings.json
// 打开方式：左下角齿轮->settings->搜索 Go: Tools Env Vars
{
    "go.goroot": "/home/damonzheng/.gvm/gos/go1.22.1",
    "go.gopath": "/home/damonzheng/.gvm/pkgsets/go1.22.1/global",
    "go.toolsEnvVars": {
        "GOPROXY": "https://goproxy.cn,direct"
    }
}
```

# 项目开发

以 [simple_go](https://github.com/NairongZheng/learning/tree/main/go_related/simple_go) 为例

**go项目结构**

```bash
simple_go/
├── cmd/                # 各个可执行程序的 main 包
│   └── simple_go/          # 主程序入口
│       └── main.go
├── internal/           # 私有逻辑包，只能被本模块导入
│   └── core/           # 核心业务逻辑
│       └── core.go
├── pkg/                # 可被外部导入的库包
│   └── utils/          # 工具函数
│       └── utils.go
├── api/                # 存放 API 定义、proto 文件等（可选）
├── configs/            # 配置文件
│   └── config.yaml
├── scripts/            # 各种脚本（如构建、部署脚本）
├── test/               # 额外测试代码
├── go.mod              # 模块定义文件
├── go.sum              # 依赖哈希校验文件
├── Makefile            # 项目构建与自动化命令
└── README.md
```

**初始化项目**

```bash
# 创建项目
mkdir simple_go
cd simple_go
# 初始化项目
go mod init github.com/<yourname>/simple_go
# 创建文件夹和文件
mkdir -p cmd/simple_go internal/core pkg/utils configs scripts test
touch cmd/simple_go/main.go internal/core/core.go pkg/utils/utils.go configs/config.yaml
```

**开发**

```go
// cmd/simple_go/main.go
package main

import (
	"fmt"
	"github.com/nairongzheng/simple_go/internal/core"
)

func main() {
	fmt.Println("Starting app...")
	core.Run()
	fmt.Println("Press Enter to exit...")
	fmt.Scanln() // 等待用户输入
}
```

```go
// internal/core/core.go
package core

import "fmt"

func Run() {
	fmt.Println("Core logic running!")
}
```

```go
// pkg/utils/utils.go
package utils

func Add(a, b int) int {
	return a + b
}
```

```go
// test/utils_test.go
package test

import (
	"fmt"
	"testing"

	"github.com/nairongzheng/simple_go/pkg/utils"
)

func TestAdd(t *testing.T) {
	result := utils.Add(1, 2)
	fmt.Println("Result of Add(1, 2):", result) // 打印结果
	t.Logf("Result of Add(1, 2) = %d", result)  // go test -v 中打印结果
	if result != 3 {
		t.Errorf("Expected 3, got %d", result)
	}
}
```

**整理依赖**

```bash
go mod tidy
```

**编写makefile**

```bash
APP_NAME = simple_go
CMD_DIR = cmd/$(APP_NAME)
BIN_DIR = bin

# 判断平台
ifeq ($(OS),Windows_NT)
	BIN_FILE = $(BIN_DIR)/$(APP_NAME).exe
	RUN_BIN = .\\$(BIN_FILE)
	CLEAN_CMD = del /Q /S $(subst /,\\,$(BIN_DIR))\* 2>NUL || exit 0 & rmdir /S /Q $(subst /,\\,$(BIN_DIR)) 2>NUL || exit 0
else
	BIN_FILE = $(BIN_DIR)/$(APP_NAME)
	RUN_BIN = ./$(BIN_FILE)
	CLEAN_CMD = rm -rf $(BIN_DIR)/
endif

.PHONY: build
build:
	go build -o $(BIN_FILE) ./$(CMD_DIR)

.PHONY: run
run:
	go run ./$(CMD_DIR)

.PHONY: run-bin
run-bin:
	$(RUN_BIN)

.PHONY: test
test:
	go test ./...

.PHONY: fmt
fmt:
	go fmt ./...

.PHONY: lint
lint:
	golangci-lint run

.PHONY: install-linter
install-linter:
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

.PHONY: clean
clean:
	$(CLEAN_CMD)
```

**运行**

```bash
make build   # 构建
make run     # 运行
make test    # 测试
```