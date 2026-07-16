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

GitLab Runner 是实际执行 CI job 的进程。项目中的 job 通过 `tags` 选择可用 runner；多个在线 runner 使用相同 tag 时，GitLab 会自动把任务交给其中有空闲容量的 runner，不保证严格轮询。

### 创建 Project Runner

在项目页面创建 runner：

```shell
Settings -> CI/CD -> Runners -> Create project runner
```

常用配置：

- Tags：填写 `.gitlab-ci.yml` 中 job 使用的 tag，例如 `project-ci`
- Run untagged：通常不勾选，避免接收未指定 tag 的任务
- Description：使用能区分机器的名称，例如 `ci-runner-node-1`

创建后会得到以 `glrt-` 开头的 authentication token。token 只在注册时使用，不要提交到仓库或写入普通日志。

### 安装 Runner

```shell
mkdir -p ~/.local/bin
curl -L -o ~/.local/bin/gitlab-runner https://gitlab-runner-downloads.s3.amazonaws.com/latest/binaries/gitlab-runner-linux-amd64
chmod +x ~/.local/bin/gitlab-runner
```

### 规划目录

单机可以直接使用默认的 `~/.gitlab-runner/config.toml`。如果多台机器共享同一个 Home 或共享存储，每台机器必须使用独立父目录：

```text
/mnt/afs_toolcall/zhengnairong/gitlab-runners/
├── node-1/
│   ├── config/config.toml
│   ├── builds/
│   ├── cache/
│   └── logs/gitlab-runner.log
└── node-2/
    ├── config/config.toml
    ├── builds/
    ├── cache/
    └── logs/gitlab-runner.log
```

`config`、`builds`、`cache` 和 `logs` 可以放在同一个父目录下，但必须使用不同子目录。不要让多台机器共用同一份 `config.toml`、`.runner_system_id`、builds 目录或日志文件。

以其中一台机器为例：

```shell
export RUNNER_ROOT="/mnt/afs_toolcall/zhengnairong/gitlab-runners/node-2"
export RUNNER_CONFIG="${RUNNER_ROOT}/config/config.toml"
export RUNNER_BUILDS="${RUNNER_ROOT}/builds"
export RUNNER_CACHE="${RUNNER_ROOT}/cache"
export RUNNER_LOG="${RUNNER_ROOT}/logs/gitlab-runner.log"

mkdir -p \
  "${RUNNER_ROOT}/config" \
  "${RUNNER_BUILDS}" \
  "${RUNNER_CACHE}" \
  "${RUNNER_ROOT}/logs"
```

### 注册 Runner

每台机器推荐在 GitLab 页面分别创建 Project Runner，获得不同的 `glrt-` token，不要复制另一台机器的 `config.toml`。

```shell
export RUNNER_TOKEN='<glrt-token>'

gitlab-runner register \
  --config "${RUNNER_CONFIG}" \
  --non-interactive \
  --url https://gitlab.xxx.com \
  --token "${RUNNER_TOKEN}" \
  --executor shell \
  --description ci-runner-node-2

unset RUNNER_TOKEN
```

注册完成后，token 会自动保存在指定的 `config.toml` 中。

注册成功时会看到类似日志：

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

### 配置并发和工作目录

编辑 `${RUNNER_CONFIG}`：

```toml
concurrent = 2
check_interval = 0
shutdown_timeout = 0

[[runners]]
  name = "ci-runner-node-2"
  url = "https://gitlab.xxx.com"
  token = "注册后自动写入，保持不变"
  executor = "shell"
  builds_dir = "/mnt/afs_toolcall/zhengnairong/gitlab-runners/node-2/builds"
  cache_dir = "/mnt/afs_toolcall/zhengnairong/gitlab-runners/node-2/cache"
  limit = 2
  request_concurrency = 2
```

参数含义：

| 参数 | 作用 |
|------|------|
| `concurrent` | 当前 runner 进程允许同时执行的 job 总数 |
| `limit` | 当前 `[[runners]]` 注册项允许同时执行的 job 数 |
| `request_concurrency` | 同时向 GitLab 请求新 job 的请求数 |
| `builds_dir` | checkout 和 job 工作目录，多机器必须隔离 |
| `cache_dir` | runner 缓存目录，多机器建议隔离 |

不要盲目提高并发。单个 job 如果会大量占用 CPU、磁盘、模型接口或网络，应该从较小值开始观察。

### 验证并启动

```shell
gitlab-runner list --config "${RUNNER_CONFIG}"
gitlab-runner verify --config "${RUNNER_CONFIG}"
```

前台启动适合调试：

```shell
gitlab-runner run \
  --config "${RUNNER_CONFIG}" \
  --working-directory "${RUNNER_ROOT}"
```

用户模式下可以后台运行：

```shell
nohup gitlab-runner run \
  --config "${RUNNER_CONFIG}" \
  --working-directory "${RUNNER_ROOT}" \
  >> "${RUNNER_LOG}" 2>&1 < /dev/null &
```

检查进程和日志：

```shell
pgrep -af gitlab-runner
tail -n 50 "${RUNNER_LOG}"
```

### 多机器自动路由

多台机器的 runner 使用相同 tag 后，GitLab 会自动调度：

```yaml
test_job:
  tags:
    - project-ci
  script:
    - python -m pytest
```

多机器运行时还需要保证：

- 每台机器都安装了 job 所需的运行环境和依赖
- 凭证、网络和数据访问权限一致
- 配置、builds、cache 和日志目录相互隔离
- 共享业务输出使用 job、pipeline 或 commit 维度的独立目录
- 不要在多个进程中直接运行同一份 runner 配置

### 常用维护命令

```shell
gitlab-runner list --config "${RUNNER_CONFIG}"
gitlab-runner verify --config "${RUNNER_CONFIG}"

# 优雅停止用户模式 runner，等待当前 job 结束
kill -QUIT <runner_pid>

# 查看日志
tail -f "${RUNNER_LOG}"
```
