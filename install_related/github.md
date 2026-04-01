
# CI/CD

更详细一点的 workflows [可以参考这个](https://github.com/NairongZheng/openclaw_gen_data/blob/main/.github/workflows/docker-image.yml)

## 什么是 CI/CD？

- CI（Continuous Integration）持续集成：代码一提交，就自动：编译、测试、检查代码
- CD（Continuous Delivery / Deployment）持续交付/部署：代码通过 CI 后，自动打包、自动发布、自动部署服务器

## GitHub Actions 的核心组件

可以把它理解为一个“任务调度系统”：

```shell
Workflow（工作流）
    ├── Job（任务）
    │    ├── Step（步骤）
    │    │     └── Action（动作）
```

对应关系：

| 概念     | 作用                           |
| -------- | ------------------------------ |
| Workflow | 整个 CI/CD 流程                |
| Job      | 一组任务（运行在一个机器上）   |
| Step     | 具体执行步骤                   |
| Action   | 封装好的功能（别人写好的插件） |


## 最小可运行例子

**在项目里创建**：

```shell
mkdir -p .github/workflows
touch .github/workflows/ci.yml
```

在其中写入：

```yml
name: My First CI

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run a script
        run: echo "Hello CI/CD!"
```

### 参数解释

**name**

```yml
# workflow 名字（随便写）
name: My First CI
```

**on（触发条件）**

```yml
# 表示只要 git push 就触发
on: [push]

# 也可以写更细：表示在 push main 分支的时候触发
on:
  push:
    branches: [main]
```

**jobs**

```yml
# 定义一个任务（名字叫 build）
jobs:
  build:
```

**runs-on**

```yml
runs-on: ubuntu-latest
# 在什么机器跑：
# ubuntu-latest（最常用）
# windows-latest
# macos-latest
```

**steps**

```yml
# 一步一步执行
steps:
```

**uses（用现成 Action）**

```yml
uses: actions/checkout@v4
# 使用 actions/checkout
# 作用：把代码 clone 到 runner
```

**run（执行命令）**

```yml
# 就是执行 shell 命令
run: echo "Hello CI/CD!"
```

## 常用核心语法

**多步骤**

```yml
steps:
  - name: Step1
    run: echo "step1"

  - name: Step2
    run: echo "step2"
```

**多 Job（并行执行）**

```yml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: echo "build"

  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "test"
```

**Job 依赖**

```yml
jobs:
  build:
    ...

  deploy:
    needs: build
# deploy 必须等 build 完成
```

**环境变量**

```yml
env:
  APP_ENV: prod

# 使用：
run: echo $APP_ENV
```

**secrets（重要！）**

```yml
# 在 GitHub 的项目设置中添加变量：Settings -> Secrets -> Actions
# 用于：服务器密码、Docker 登录、API Key
env:
  TOKEN: ${{ secrets.MY_TOKEN }}
```



