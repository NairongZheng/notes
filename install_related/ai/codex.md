- [环境配置](#环境配置)
  - [安装和更新](#安装和更新)
  - [登录（openai）](#登录openai)
  - [第三方 api](#第三方-api)
  - [配置\&使用](#配置使用)
    - [config.toml](#configtoml)
    - [Codex AGENTS.md](#codex-agentsmd)
    - [skill](#skill)
- [Linux 使用](#linux-使用)
  - [本机终端](#本机终端)
- [常用终端操作](#常用终端操作)
- [macOS App 使用](#macos-app-使用)
  - [连接远程 SSH 主机](#连接远程-ssh-主机)


Codex 官方文档：[https://developers.openai.com/codex/](https://developers.openai.com/codex/)

# 环境配置

## 安装和更新

**使用 npm 安装（Linux、macOS 通用）**

```shell
# 需要提前安装 Node.js 和 npm
npm install -g @openai/codex

# 更新到最新版本
npm install -g @openai/codex@latest

# 检查是否安装成功
codex --version
```

**使用 Homebrew 安装（macOS）**

```shell
brew install --cask codex

# 更新
brew upgrade codex
```

## 登录（openai）

Codex CLI 支持使用 ChatGPT 账号或 OpenAI API Key。个人交互使用一般直接登录 ChatGPT；脚本和 CI 更适合使用 API Key。

```shell
# linux 直接输入 codex 就有引导了

# 普通电脑：会拉起浏览器完成登录
codex login

# 无桌面的 Linux / SSH 服务器：使用设备码登录（beta）
codex login --device-auth

# 检查当前登录状态
codex login status

# 退出登录
codex logout
```

## 第三方 api

```shell
# 使用第三方直接先写最小配置，注意 codex 用的是 /v1/responses
# vim ~/.codex/config.toml

model = "你的模型名"
model_provider = "thirdparty"

[model_providers.thirdparty]
name = "Third Party such as openrouter"
base_url = "https://你的服务地址/v1"
env_key = "OURNROUTER_API_KEY"
wire_api = "responses"

# 可以把 key 写进环境变量
# export OURNROUTER_API_KEY="sk-or-v1-xxx"
```

## 配置&使用

### config.toml

配置文件常见位置：

```shell
~/.codex/config.toml            # 个人全局配置
<project>/.codex/config.toml    # 项目配置
codex -c key=value              # 临时覆盖
# 优先级是：命令行参数 > 项目配置 > Profile > 个人配置 > Codex 默认值。
```

配置文件主要还是用 `~/.codex/config.toml`。如果只想临时试一下，也可以直接在启动命令里覆盖：

```shell
codex --model gpt-5.6
codex --profile fast
codex --config model_reasoning_effort='"high"'
```

补充：

- 项目内的 `.codex/config.toml` 只有在该目录**被标记为 trusted 后才会生效**。
- 想长期生效就改 `~/.codex/config.toml`；想只试一次就用启动参数。
- app、CLI、IDE extension 共用同一套 `config.toml` 层级；但命令行临时参数只影响当前这一次启动。

### Codex AGENTS.md

Codex 每次启动时会读取 `AGENTS.md`。全局规则先加载，项目目录中越靠近当前工作目录的规则越后加载，因此可以覆盖上层规则。

```shell
~/.codex/AGENTS.md              # 个人全局偏好
<project>/AGENTS.md             # 项目规则，建议提交到仓库
<project>/src/AGENTS.md         # src 目录及其子目录的补充规则
<project>/src/AGENTS.override.md # 临时覆盖同目录的 AGENTS.md
```

可以使用 `/init` 在当前项目生成初始文件。示例：

```markdown
# General

- 默认使用中文回答。
- 修改代码前先简单说明计划。
- 每次需求的添加不要都创建一个冗余的文档，可以给予原来的文档进行调整，如有必要再新增。
- 优先修改最少的代码。
- 单一代码文件不能太长，需要做好模块化，但是不要无意义重构。
- 不要修改无关文件。
- 新增依赖前先询问。
- 如果需要运行 python 脚本，可以使用 conda 下的 dev 环境。

# Coding Style

- Python 使用 type hints。
- 函数尽量保持简短。
- 函数需要写注释，函数内部关键逻辑也需要添加注释。
- 优先使用标准库。

# Git

- commit 信息使用英文和 Conventional Commits 格式，如 `feat: xxx`。
- 不主动 commit。
- 不主动 push。

# Testing

- 修改代码后运行相关测试，并说明验证结果。
```

规则较多时，可以把通用约定放在项目根目录，把某个模块特有的规则放到对应子目录，不要把所有细节都堆在一个文件中。

### skill

**skill 格式**

```shell
my-skill/
├── SKILL.md
├── scripts/          # 可选脚本
├── references/       # 可选参考资料
├── assets/           # 可选图片、模板
└── agents/
    └── openai.yaml   # 可选 App 展示信息
```

其中 SKILL.md 格式是：

```shell
---
name: my-skill
description: 当用户需要检查项目结构和质量时使用。
---

# 项目检查 Skill

1. 阅读项目目录。
2. 检查 Git 状态。
3. 查找测试与构建命令。
4. 输出中文检查报告。
```

**安装 skill**

```shell
~/.codex/skills/.system/    # 系统内置，不要修改
~/.agents/skills/           # 个人全局 skill，可以直接复制到这里
<project>/.agents/skills/   # 项目 skill，适合提交到项目仓库
~/.codex/plugins/cache/     # Plugin 提供，由 Plugin 管理，不要直接修改
```

**查看 skill**

```shell
# 1. app 对话框
$skill_name
# 2. app 界面 -> plugins -> skills
# 3. cli
/skills
```

# Linux 使用

## 本机终端

进入需要操作的项目目录后启动 Codex：

```shell
cd <your_project>

# 启动交互式终端
codex

# 启动时直接给出任务
codex "检查当前项目，并说明如何运行测试"

# 查看命令帮助
codex --help
```

第一次在目录中使用时，需要确认是否信任该目录。默认的 `Auto` 权限允许 Codex 在当前工作区内读写和运行命令；涉及工作区外部或网络访问时仍会请求确认。

# 常用终端操作

```shell
# 恢复历史会话；不带参数时打开选择器
codex resume

# 恢复最近一次会话
codex resume --last

# 在指定目录启动
codex --cd <your_project>

# 使用指定模型启动
codex --model <model_name>
```

交互会话中常用的斜杠命令：

```shell
/plan          # 先梳理需求和执行计划；也可以使用 Shift+Tab 切换
/permissions   # 切换 Auto、Read-only、Full Access 等权限
/status        # 查看模型、权限、工作区和上下文状态
/model         # 切换模型
/init          # 在当前项目生成 AGENTS.md 初始文件
/review        # 检查当前工作区的代码修改
/compact       # 压缩当前会话上下文
/new           # 在当前 CLI 中开始一个新会话
/resume        # 恢复已保存的会话
/quit          # 退出
```

建议复杂任务先进入 `/plan`，明确目标、相关文件、约束和完成标准后再开始修改。

# macOS App 使用

Codex App 适合同时管理多个项目或任务，支持本地线程、Git worktree、diff 查看、commit/push 和内置终端。

1. 从 [Codex App 官方页面](https://developers.openai.com/codex/app/) 下载 macOS 版本；Intel Mac 选择 Intel 构建。
2. 使用 ChatGPT 账号或 OpenAI API Key 登录。API Key 登录时，部分依赖 ChatGPT 工作区或云端的功能可能不可用。
3. 添加项目目录，新建线程时选择运行方式：
   - `Local`：直接修改当前项目目录，适合单任务。
   - `Worktree`：为线程创建独立 Git worktree，适合并行任务或试验性修改。
   - `Cloud`：在已经配置的云端环境运行。
4. 在右侧 diff 面板检查修改；可以逐文件或逐块 stage/revert，也可以直接 commit、push 或创建 PR。
5. 使用右上角终端按钮或 `Cmd+J` 打开当前项目/工作树对应的内置终端。

简单使用时，选择项目后保持 `Local` 即可。并行开发时优先用 `Worktree`，避免多个线程同时修改同一个工作目录。

## 连接远程 SSH 主机

SSH 别名已经配置好时，只需：

1. 打开 Codex App 的 `Settings > Connections`。
2. 添加或启用对应主机，例如 `tj`。
3. 重启 App，然后选择该主机上的项目目录。
