- [环境配置](#环境配置)
  - [安装和更新](#安装和更新)
- [使用](#使用)
  - [Profile 概念](#profile-概念)
  - [Web GUI](#web-gui)
  - [TUI（终端交互）](#tui终端交互)
  - [Headless（一次性问答）](#headless一次性问答)
  - [常用命令](#常用命令)
- [AGENTS.md 长期记忆](#agentsmd-长期记忆)
- [插件管理](#插件管理)
- [上下文压缩](#上下文压缩)


DeepSeek Harness（简称 dsh）是一个基于 profile + 插件组合的 agent 框架，同一套内核可以启动成 Web GUI、终端 TUI 或一次性 headless 问答。

# 环境配置

## 安装和更新

**使用 npm 安装（需要提前装好 Node.js 和 npm）**

```shell
# 全局安装（推荐，命令稳定可用）
npm install -g @deepseek-ai/dsh

# 安装指定版本（当前预发布版本为例）
npm install -g @deepseek-ai/dsh@0.1.5-rc.2

# 更新到最新版
npm install -g @deepseek-ai/dsh@latest

# 检查是否安装成功
dsh --version
```

**不安装、临时用 npx 拉起（每次可能检查/下载）**

```shell
npx @deepseek-ai/dsh@0.1.5-rc.2 web

# 注意：npx 只把命令临时放进缓存目录（~/.npm/_npx/xxx/node_modules/.bin）
# 缓存可能被清理，新开终端可能出现 "dsh not found"，长期使用建议用上面的全局安装
```

配置目录默认在 `~/.dsh`（即环境变量 `$DSH_HOME`），主要内容：

```shell
~/.dsh/settings.yaml      # 全局设置：模型提供方、默认模型、权限等
~/.dsh/AGENTS.md          # 用户全局记忆（见下文）
~/.dsh/profiles/          # 各个 profile（web / tui 等）
~/.dsh/sessions/          # 会话历史
~/.dsh/.credentials.yaml  # 凭据
```

# 使用

## Profile 概念

dsh 通过 `--profile <名字>` 启动不同形态的应用，profile 存放在 `~/.dsh/profiles/`。

```shell
# 首次使用某个 profile 前，需要从内置模板派生一份
dsh --profile tui --from-default-profile tui

# 之后直接启动即可
dsh --profile tui
```

内置常见的形态有 `web`、`tui`、`headless` 三种，下面分别介绍。

## Web GUI

浏览器界面，功能最全。

```shell
dsh web
# 等价于 dsh --profile web
# 启动后按提示打开本地地址（如 http://127.0.0.1:3080）
```

## TUI（终端交互）

在终端里的全屏交互界面，适合纯命令行环境。

```shell
# 首次需要先派生 profile
dsh --profile tui --from-default-profile tui

# 启动
dsh --profile tui
```

## Headless（一次性问答）

给一个任务、回答完就退出，最适合写进脚本或 CI。

```shell
dsh --profile headless "帮我跑一下测试并说明结果"
```

## 常用命令

```shell
dsh --version                          # 版本
dsh --profile <name> --help            # 查看某个 profile（app）自己的参数
dsh --profile <name> --resume <session> # 恢复指定会话
dsh --dump-config                      # 打印当前 profile 合成后的插件树
dsh --patch ./extra.yml --profile tui  # 在 profile 之上叠加一层临时配置
```

# AGENTS.md 长期记忆

dsh 通过 `dsh-agent-instructions` 插件加载 `AGENTS.md` 风格的指令文件作为长期记忆，`dsh-base` 默认已开启。加载顺序为“从宽到窄”，越靠下越具体、优先级越高：

```shell
~/.dsh/AGENTS.md              # 用户全局偏好，跨所有项目生效
<project>/AGENTS.md           # 项目规则，建议提交到仓库
<project>/CLAUDE.md           # 同上，Claude 兼容名（内容重复会自动去重）
<project>/子目录/AGENTS.md     # 该目录及子目录的补充规则
<project>/AGENTS.local.md     # 个人私有覆盖层，建议加进 .gitignore
```

- 项目根通过 `.git` 目录识别；从根往下每层目录的 `AGENTS.md` 都会叠加。
- 总注入预算 65536 字节，超出时先丢弃更宽泛的文件、最后才截断最具体的那个。
- **AGENTS.md 是静态指令，agent 不会自动往里写**，想让某条偏好长期生效需要手动写入。

示例内容：

```markdown
# General

- 默认使用中文回答。
- 修改代码前先简单说明计划。
- 优先修改最少的代码，不要修改无关文件。
- 单一代码文件不能太长，需要做好模块化，但不要无意义重构。
- 新增依赖前先询问。

# Coding Style

- Python 使用 type hints，函数尽量简短，关键逻辑加中文注释。
- 优先使用标准库。

# Git

- commit 信息使用英文和 Conventional Commits 格式，如 `feat: xxx`。
- 不主动 commit，不主动 push。
```


# 插件管理

插件通过 `dsh plugin` 命令安装到指定 profile：

```shell
# 给 tui profile 安装插件（本质是转发给 pnpm）
dsh plugin --profile tui add <包名>

# 可以先安装“插件市场”插件，后面就方便了
dsh plugin --profile web add dshmarket
```

一些插件网站：
- https://github.com/awesome-dsh-plugin/awesome-dsh-plugin
- https://dshplugin.online/plugins

# 上下文压缩

长对话由 `dsh-compaction-basic` 插件处理，`dsh-base` 默认已开启：

- 对话接近模型上下文上限时**自动压缩**最旧的历史为摘要，保留近期消息。
- 触发上下文溢出错误后，会自动压缩并重试。
- 也可以在对话中**手动压缩**：

```shell
/compact    # 立即把较旧历史压缩为一条摘要，显示压缩条数和节省的 token
```
