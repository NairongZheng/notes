
# Github

## Actions

**什么是 CI/CD？**

- CI（Continuous Integration）持续集成：代码一提交，就自动：编译、测试、检查代码
- CD（Continuous Delivery / Deployment）持续交付/部署：代码通过 CI 后，自动打包、自动发布、自动部署服务器

**Github Actions 的核心组件**

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

**示例**

```yml
# 文件路径：<repo_root>/.github/workflows/ci.yml

# workflow 名字：显示在 GitHub Actions 页面中
name: Demo CI Pipeline

# on 触发条件：
# 1. pull_request: pr 时运行
# 2. push: push 到指定分支时自动运行
# 3. workflow_dispatch: 在 GitHub 页面里手动点击 Run workflow 时运行
on:
  pull_request:
  push:
    branches:
      - main
      - master
  workflow_dispatch:
    inputs: # 输入的参数，在 github 页面运行时会有一个 ui
      param1:
        description: xxx
        required: true  # 是否必须，true 或者 false
        default: xxx    # 默认值
        type: boolean   # 类型，boolean 会出现是否勾选的框

# 全局环境变量：所有 job 都能直接使用
env:
  APP_ENV: prod
  TOKEN: ${{ secrets.MY_TOKEN }}  # secrets 用来放敏感信息，比如 token、密码、API key，这些值不要硬编码进仓库，而是在仓库设置中配置

# jobs 表示整条 workflow 里要执行哪些任务
jobs:
  build:
    # runs-on 指定 runner 环境
    # 常见值：ubuntu-latest / windows-latest / macos-latest
    runs-on: ubuntu-latest

    # steps 表示在当前 job 里按顺序执行的步骤，一个 job 里可以连续写很多 step
    steps:
      - name: Checkout code
        # uses 表示复用别人已经写好的 action
        # actions/checkout 的作用：把仓库代码拉到 runner 机器上
        uses: actions/checkout@v4

      - name: Print current environment
        # run 表示直接执行 shell 命令
        run: echo $APP_ENV

      - name: Build project
        run: echo "build"

  test:
    # 如果没有 needs，通常可以和别的 job 并行执行
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Run tests
        run: echo "test"

  deploy:
    # needs 表示依赖关系，deploy 必须等 build 完成后才会执行
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy project
        run: echo "deploy"
```

更详细一点的 workflows [可以参考这个](https://github.com/NairongZheng/openclaw_gen_data/blob/main/.github/workflows/docker-image.yml)


**Github Actions 配置与运行**

配置变量：

```shell
# CI 中可能需要用到一些变量，比如 api key，需要在 run workflows 之前设置好：
# Settings -> Secrets and variables -> Repository secrets -> Actions
# 然后依次填好需要的变量
```

Run workflow：

```shell
# Actions -> 选择要跑的 workflow -> Run workflow -> 填写 inputs -> Run
```

# Gialab

## Pipelines

**与 Github 的 Actions 对比**

> Gitlab 的 CI/CD 和 Github 本质上做的是同一件事：
> 
> - CI：代码提交后自动构建、自动测试
> - CD：构建通过后自动部署
> 
> 但 Gitlab 和 Github 的一个很大区别是：
> 
> - Github 是通过 Github Actions 来做 CI/CD
> - Gitlab 是平台内置了 CI/CD 体系

**Gitlab Pipelines 的核心组件**

```shell
Pipeline（流水线）
    ├── Stage（阶段）
    │     ├── Job（任务）
```

对比 GitHub：

| GitHub Actions | GitLab CI/CD   |
| -------------- | -------------- |
| Workflow       | Pipeline       |
| Job            | Job            |
| Step           | script（命令） |
| 无明确 Stage   | Stage（阶段）  |

关键差异：
- GitHub 更常通过 `needs` 表达任务依赖
- GitLab 天然强调 `stages`，先按阶段分层，再在同一阶段内并行执行 job

**例子**

```yml
# 文件路径：<repo_root>/.gitlab-ci.yml

# workflow: rules 是“pipeline 级别”的规则
# 它控制的是：这次提交 / 操作，要不要创建整条 pipeline
# 如果这里没有匹配到，后面的 stages 和 jobs 根本不会开始跑
workflow:
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'    # merge_request_event：创建或更新 Merge Request 时触发 pipeline
    - if: '$CI_COMMIT_BRANCH == "main"'     # 当提交发生在 main 分支时，也创建 pipeline
    - if: '$CI_COMMIT_BRANCH == "master"'   # 有些仓库默认分支还叫 master，也一起放行
    - if: '$CI_PIPELINE_SOURCE == "web"'    # web 表示在 GitLab 页面手动点击 Run pipeline 时触发
    - when: never       # 最后一条通常会显式写 never，意思是：上面都不匹配时，不创建 pipeline

# image 指定 CI 运行时使用的 Docker 镜像
# 可以理解成：这条 pipeline 在什么环境里执行
image: python:3.10

# variables 定义全局环境变量，后续所有 job 都可以直接读取
# gitlab 没有像 github 可以在 workflow_dispatch 的 inputs 有自动的很好看的 ui，不过也是可以在这边设置一些默认值
variables:
  APP_ENV: prod
  PARAM1:
    description: "xxx"  # 描述
    value: "xxx"        # 默认值
    options:            # 选项，类似 github workflow 中的 boolean 类型
      - "true"
      - "false"


# stages 定义流水线阶段顺序
# Gitlab 会严格按这里的顺序推进：先 build，再 test，最后 deploy
stages:
  - build
  - test
  - deploy

# before_script：所有 job 执行前统一先跑的命令
before_script:
  - echo "before"

# after_script：所有 job 执行后统一跑的命令
after_script:
  - echo "after"

# cache：缓存目录，用来加速构建
# 这里以 node_modules 为例，实际项目里可以换成你自己的依赖目录
cache:
  paths:
    - node_modules/

# build_job：一个最基础的 job
build_job:
  # stage 指定当前 job 属于哪个阶段
  stage: build
  script:
    # script 就是当前 job 要执行的命令列表
    - echo $APP_ENV
    - echo "hello" > output.txt
    - echo "build"
  # artifacts：保存构建产物，供后续 job 或下载使用
  artifacts:
    paths:
      - output.txt

# test_job：放在 test 阶段
test_job:
  stage: test
  script:
    - python --version
    - echo "test"

# deploy_job：放在 deploy 阶段
deploy_job:
  stage: deploy
  # 这里的 rules 是“job 级别”的规则
  # 和上面的 workflow: rules 不同：
  # - job rules 决定 pipeline 创建后，这个 job 要不要执行
  # 这里表示只有当前分支是 main 才会执行部署 job
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
  script:
    - echo "deploy"
```

更详细一点的 pipelines [可以参考这个](https://github.com/NairongZheng/openclaw_gen_data/blob/main/.gitlab-ci.yml)

**Gitlab pipelines 配置与运行**

配置变量：

```shell
# settings -> CI/CD -> Variables
# 然后依次填好需要的变量
```

Run pipeline：

```shell
# Bulids -> New pipelines -> 配置好变量（可以覆盖上面配置的）-> Run pipeline
```

**github 与 gitlab 的 CI/CD 对比**

| 对比项      | GitHub Actions          | GitLab CI/CD            |
| ----------- | ----------------------- | ----------------------- |
| 配置文件    | .github/workflows/*.yml | .gitlab-ci.yml          |
| 执行单位    | Workflow                | Pipeline                |
| 阶段控制    | needs                   | stages                  |
| 执行方式    | step-by-step            | job script              |
| Runner      | GitHub 提供             | GitLab Runner（需配置） |
| Docker 支持 | 可选                    | 默认强依赖              |
| 上手难度    | 简单                    | 稍复杂但更强大          |