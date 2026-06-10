

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

**进入 codex**

```shell
# 可以先查看命令的使用方式
codex -h

# 直接进入
codex

# 进入某个回话
codex resume
```

**一些命令**

```shell
# 状态检查
/status

# 权限管理
/approvals  # 里面有个 read-only 可以当作类似 claudecode 的 plan 模式使用
```
