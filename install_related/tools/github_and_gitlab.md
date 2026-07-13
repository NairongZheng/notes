# Github

## Actions

**什么是 CI/CD？**

- CI（持续集成）：代码一提交，就自动编译、测试、检查代码
- CD（持续交付/部署）：代码通过 CI 后，自动打包、自动发布、自动部署到服务器

**核心组件**

可以把它理解为一个"任务调度系统"，从大到小一共四层：

```shell
Workflow（工作流）
    ├── Job（任务）
    │    ├── Step（步骤）
    │    │     └── Action（动作）
```

| 概念 | 作用 |
|------|------|
| Workflow 工作流 | 整个 CI/CD 流程，定义在 `.github/workflows/*.yml` 里 |
| Job 任务 | 运行在一台机器上的一组步骤（多个 Job 默认并行） |
| Step 步骤 | 每一步操作：要么 `uses` 用别人写好的动作，要么 `run` 直接跑命令 |
| Action 动作 | 封装好的可复用功能，别人写好的"插件" |

**示例**

```yml
# 文件路径：<repo_root>/.github/workflows/ci.yml

# workflow 名字（显示在 GitHub Actions 页面）
name: 示例流水线

# on 触发条件：
# 1. pull_request: 创建/更新 PR 时运行
# 2. push: 推送到指定分支时运行
# 3. workflow_dispatch: 在 GitHub 页面手动点 Run workflow 运行
on:
  pull_request:
  push:
    branches:
      - main
      - master
  workflow_dispatch:
    inputs: # 手动运行时的输入参数（会在页面上生成 UI）
      param1:
        description: "这里是参数描述"
        required: true  # 是否必须，true 或者 false
        default: xxx    # 默认值
        type: boolean   # 类型，boolean 会出现是否勾选的框

# 全局环境变量（所有 job 都能用）
env:
  APP_ENV: prod
  # 敏感信息（token、密码、API key）不要在代码里硬编码
  # 在 Settings → Secrets and variables → Actions 中配置，使用时：${{ secrets.变量名 }}
  TOKEN: ${{ secrets.MY_TOKEN }}

# jobs 整条 workflow 里要执行的任务
jobs:
  build:
    # runs-on 指定运行环境：ubuntu-latest / windows-latest / macos-latest
    runs-on: ubuntu-latest

    # steps 表示在当前 job 里按顺序执行的步骤，一个 job 里可以连续写很多 step
    steps:
      - name: 拉取代码
        # uses 表示复用别人写好的 action
        # actions/checkout 的作用是把仓库代码拉到 runner 机器上
        uses: actions/checkout@v4

      - name: 输出环境变量
        # run 表示直接执行 shell 命令
        run: echo $APP_ENV

      - name: 构建项目
        run: echo "构建完成"

  test:
    # 不写 needs 则默认和其他 job 并行执行
    runs-on: ubuntu-latest
    steps:
      - name: 拉取代码
        uses: actions/checkout@v4

      - name: 运行测试
        run: echo "测试通过"

  deploy:
    # needs 表示依赖：deploy 必须等 build 完成后再执行
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: 部署项目
        run: echo "部署完成"
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

# Gitlab

## Pipelines

**Gitlab Pipelines 的核心组件**

```shell
Pipeline（流水线）
    ├── Stage（阶段）
    │     ├── Job（任务）
```

| 概念 | 作用 |
|------|------|
| Pipeline 流水线 | 整个 CI/CD 流程，定义在 `.gitlab-ci.yml` 里 |
| Stage 阶段 | 流水线的一个阶段（如构建、测试、部署），同一阶段内的 Job 并行执行 |
| Job 任务 | 阶段中的执行单元，定义具体要跑的命令 |

对比 GitHub：

| GitHub Actions | GitLab CI/CD |
|---------------|-------------|
| Workflow | Pipeline |
| 无明确 Stage | Stage（阶段） |
| Job | Job |
| Step | script（命令） |

关键差异：
- GitHub 更常用 `needs` 表达任务依赖
- GitLab 天然强调 `stages`，先按阶段分层，再在同一阶段内并行执行 job

**示例**

```yml
# 文件路径：<repo_root>/.gitlab-ci.yml

# workflow rules: pipeline 级别的规则
# 控制"这次提交要不要创建 pipeline"
# 这里不匹配的话，后面的 stages 和 jobs 根本不会开始跑
workflow:
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'   # 创建/更新 MR 时触发
    - if: '$CI_COMMIT_BRANCH == "main"'                     # 推送到 main 分支时触发
    - if: '$CI_COMMIT_BRANCH == "master"'                   # 也兼容旧仓库的 master 分支
    - if: '$CI_PIPELINE_SOURCE == "web"'                    # 在页面上手动点 Run pipeline 时触发
    - when: never       # 上面都不匹配时，不创建 pipeline

# image: CI 运行时的 Docker 镜像
image: python:3.10

# variables: 全局环境变量，所有 job 都可以直接读取
# 不像 github 在 workflow_dispatch inputs 有自动的 UI，
# 但可以在这边设置一些默认值和选项
variables:
  APP_ENV: prod
  PARAM1:
    description: "这里是参数描述"
    value: "xxx"        # 默认值
    options:            # 限定可选值，类似 github workflow 的 boolean
      - "true"
      - "false"

# stages: 定义流水线阶段顺序
# Gitlab 会严格按这里的顺序推进：先 build，再 test，最后 deploy
stages:
  - build
  - test
  - deploy

# before_script：所有 job 执行前统一跑的命令
before_script:
  - echo "准备工作..."

# after_script：所有 job 执行后统一跑的命令
after_script:
  - echo "清理工作..."

# cache：缓存目录，加速后续构建
cache:
  paths:
    - node_modules/

# build_job：最基础的 job
build_job:
  # stage 指定当前 job 属于哪个阶段
  stage: build
  script:
    # script 就是当前 job 要执行的命令列表
    - echo $APP_ENV
    - echo "构建结果" > output.txt
    - echo "构建完成"
  # artifacts：保存构建产物，供后续 job 使用或从页面下载
  artifacts:
    paths:
      - output.txt

# test_job：放在 test 阶段
test_job:
  stage: test
  script:
    - python --version
    - echo "测试通过"

# deploy_job：放在 deploy 阶段
deploy_job:
  stage: deploy
  # 这里的 rules 是“job 级别”的规则
  # 和上面的 workflow: rules 不同：
  # job rules 决定 pipeline 创建后，这个 job 要不要执行
  # 这里表示只有当前分支是 main 才会执行部署 job
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
  script:
    - echo "部署完成"
```

更详细的 pipelines [可以参考这个](https://github.com/NairongZheng/openclaw_gen_data/blob/main/.gitlab-ci.yml)

**Gitlab pipelines 配置与运行**

配置变量：

```shell
# settings -> CI/CD -> Variables
# 依次填好需要的变量
```

Run pipeline：

```shell
# Builds -> Pipelines -> Run pipeline -> 配置变量（可以覆盖上面配置的） -> Run
```

**github 与 gitlab 的 CI/CD 对比**

| 对比项 | GitHub Actions | GitLab CI/CD |
|--------|---------------|-------------|
| 配置文件 | .github/workflows/*.yml | .gitlab-ci.yml |
| 执行单位 | Workflow | Pipeline |
| 阶段控制 | needs | stages |
| 执行方式 | step-by-step（uses / run） | job script |
| Runner | GitHub 提供 | GitLab Runner（需配置） |
| Docker 支持 | 可选 | 默认强依赖 |
| 上手难度 | 简单 | 稍复杂但更强大 |

## glab cli

**登录**

```shell
(dev) ➜  /mnt/afs_toolcall/zhengnairong git:(main) glab auth login
? What GitLab instance do you want to log into? GitLab Self-hosted Instance
? GitLab hostname: gitlab.xxx.com
? API hostname: gitlab.xxx.com
- Logging into gitlab.xxx.com
? How would you like to login? Token

Tip: you can generate a Personal Access Token here https://gitlab.xxx.com/-/profile/personal_access_tokens?scopes=api,write_repository
The minimum required scopes are 'api' and 'write_repository'.
? Paste your authentication token: ********************
? Choose default git protocol HTTPS
? Authenticate Git with your GitLab credentials? Yes
? Choose host API protocol HTTPS
- glab config set -h gitlab.xxx.com git_protocol https
✓ Configured git protocol
- glab config set -h gitlab.xxx.com api_protocol https
✓ Configured API protocol
✓ Logged in as zhengnairong
```

**查看登陆状态**

```shell
(dev) ➜  /mnt/afs_toolcall/zhengnairong git:(main) glab auth status
gitlab.com
  x gitlab.com: api call failed: GET https://gitlab.com/api/v4/user: 401 {message: 401 Unauthorized}
  ✓ Git operations for gitlab.com configured to use ssh protocol.
  ✓ API calls for gitlab.com are made over https protocol
  ✓ REST API Endpoint: https://gitlab.com/api/v4/
  ✓ GraphQL Endpoint: https://gitlab.com/api/graphql/
  x No token provided
gitlab.xxx.com
  ✓ Logged in to gitlab.xxx.com as zhengnairong (/mnt/xxx/zhengnairong/.config/glab-cli/config.yml)
  ✓ Git operations for gitlab.xxx.com configured to use https protocol.
  ✓ API calls for gitlab.xxx.com are made over https protocol
  ✓ REST API Endpoint: https://gitlab.xxx.com/api/v4/
  ✓ GraphQL Endpoint: https://gitlab.xxx.com/api/graphql/
  ✓ Token: **************************
```

### glab mr

**常用命令**

```shell
# 查看 MR 列表
glab mr list

# 查看 MR 详情
glab mr view <MR编号>

# 在浏览器中打开
glab mr view <MR编号> --web

# 创建 MR
glab mr create

# 根据当前分支信息快速创建
glab mr create --fill

# 拉取并切换到 MR 分支
glab mr checkout <MR编号>

# 查看代码差异
glab mr diff <MR编号>

# 添加评论
glab mr note <MR编号> -m "评论内容"

# 更新标题、描述、标签等
glab mr update <MR编号>

# 批准 MR
glab mr approve <MR编号>

# 撤销批准
glab mr revoke <MR编号>

# 合并 MR
glab mr merge <MR编号>

# 触发 rebase
glab mr rebase <MR编号>

# 关闭 / 重新打开
glab mr close <MR编号>
glab mr reopen <MR编号>

# 删除 MR
glab mr delete <MR编号>
```


## runner

**配置 runner**

```shell
# Settings -> CI/CD -> Runners -> New project runner
# tag 填一下
```

**安装部署 runner**

```shell
mkdir -p ~/.local/bin
curl -L -o ~/.local/bin/gitlab-runner https://gitlab-runner-downloads.s3.amazonaws.com/latest/binaries/gitlab-runner-linux-amd64
chmod +x ~/.local/bin/gitlab-runner

gitlab-runner register \
  --url https://gitlab.xxx.com \
  --token '<runner token>' \
  --executor shell \
  --name tj-doublecheck-runner
```

成功会有 log：

```shell
gitlab-runner register \
  --url https://gitlab.xxx.com \
  --token xxx \
  --executor shell \
  --name tj-doublecheck-runner
Runtime platform                                    arch=amd64 os=linux pid=412228 revision=24b9b726 version=19.1.1
WARNING: Running in user-mode.
WARNING: The user-mode requires you to manually start builds processing:
WARNING: $ gitlab-runner run
WARNING: Use sudo for system-mode:
WARNING: $ sudo gitlab-runner...

Enter the GitLab instance URL (for example, https://gitlab.com/):
[https://gitlab.xxx.com]:
Verifying runner... is valid                        correlation_id=01KX5Q8F4EM3960RMNPZFWW1MK runner=yRy8Skv5R runner_name=tj-doublecheck-runner
Enter a name for the runner. This is stored only in the local config.toml file:
[tj-doublecheck-runner]:
Enter an executor: instance, docker, docker-windows, docker+machine, docker-autoscaler, kubernetes, ssh, parallels, virtualbox, shell, custom:
[shell]:
Runner registered successfully. Feel free to start it, but if it's running already the config should be automatically reloaded!

Configuration (with the authentication token) was saved in "/mnt/afs_toolcall/zhengnairong/.gitlab-runner/config.toml"
```

**运行 runner**

```shell
gitlab-runner run
```