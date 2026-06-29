

# 环境配置

**安装**

```shell
# 安装方式1:
npm install -g @openai/codex

# 安装方式2(macOS):
brew install codex
```

**更新**

```shell
# 安装方式1:
npm install -g @openai/codex@latest

# 安装方式2(macOS):
brew upgrade codex
```

# 使用

## 终端使用

**登陆**

```shell
codex login --device-auth
```

**进入 codex**

```shell
# 可以先查看命令的使用方式
codex -h

# 直接进入
codex

# 进入某个会话
codex resume
```

# codex AGENTS

```shell
~/.codex/AGENTS.md      # 全局偏好
项目根目录/AGENTS.md      # 项目规则
src/AGENTS.md           # 子目录规则
```

示例 AGENTS.md

```shell
# General

- 默认使用中文回答。
- 修改代码前先简单说明计划。
- 优先修改最少的代码。
- 不要无意义重构。
- 不要修改无关文件。
- 新增依赖前先询问。

# Coding Style

- Python 使用 type hints。
- 函数尽量保持简短。
- 函数需要写注释
- 优先标准库。

# Git

- commit 信息要英文，用标准的格式，如 `feat: xxx` 等，可以分点。
- 不主动 commit。
- 不主动 push。

# Testing

- 修改代码后说明应该如何验证。
```