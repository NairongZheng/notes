- [安装](#安装)
- [与 openclaw 交互](#与-openclaw-交互)
- [初始化设置](#初始化设置)
  - [一键设置](#一键设置)
  - [网关](#网关)
  - [模型配置](#模型配置)
  - [channels 配置](#channels-配置)
- [使用](#使用)
  - [会话管理](#会话管理)
  - [agent 管理](#agent-管理)
- [基本命令](#基本命令)
- [目录结构](#目录结构)
- [记忆系统](#记忆系统)
- [常见问题](#常见问题)
- [进阶技巧](#进阶技巧)
- [资源链接](#资源链接)


> 开源的个人 AI 助手平台，运行在你自己的设备上
> 
> 官网: https://openclaw.ai/
> 
> 中文文档: https://openclawcn.cn/docs.html
> 
> 社区: https://discord.com/invite/clawd


## 安装

```bash
# macOS / Linux 一键安装
curl -fsSL https://openclaw.ai/install.sh | bash

# 重启终端或运行
source ~/.zshrc  # 或 source ~/.bashrc

# 检查版本
openclaw --version
```

## 与 openclaw 交互

与 openclaw 连接交互有以下方式（要先配好网关）：
1. terminal：`openclaw tui --url ws://127.0.0.1:<gateway_port> --token <gateway_token>`
2. browser：`http:127.0.0.1:<gateway_port>`，然后到“控制-概览”填写网关 token
3. 配置频道，使用不同 app 进行交互。

先看看下面的初始化设置

## 初始化设置

### 一键设置

```shell
# 运行安装向导（关键！！）
openclaw onboard --install-daemon
# 第一次初始化的话 Onboarding mode 可以选择 Manual 模式进行详细配置
```

### 网关

```shell
openclaw gateway --port 18789 --verbose
# 一些网关命令
openclaw gateway install   # 安装为系统服务
openclaw gateway start     # 启动
openclaw gateway status    # 查看状态
openclaw gateway stop      # 停止
openclaw gateway restart   # 重启
```

### 模型配置

```bash
openclaw configure
# 选择 `Model` -> `Custom Provider`，然后填写：
# 1. API Base URL（如 `https://api.anthropic.com/v1`）
# 2. API Key
# 3. 类型（OpenAI / Anthropic / Other）
# 4. Model ID（如 `claude-sonnet-4-5-20250929`）
# OpenClaw 对未知模型默认限制为 4096 tokens，需手动修改 `contextWindow`。

# 查看模型配置
openclaw models
```

### channels 配置

```shell
# 一些参考命令
openclaw channels list
openclaw channels status
```

**telegram 配置**

1. 找 [@BotFather](https://t.me/BotFather) 创建机器人
2. 发送 `/newbot` 并按提示操作
3. 获取 Bot Token
4. 运行 `openclaw configure` → `Channels` → `Telegram`
5. 输入 Token
6. 在 Telegram 中搜索你的机器人并发送 `/start`

**首次配对**
第一次发送消息会收到配对请求，在终端运行：
```bash
openclaw pairing approve telegram <配对码>
```

**配对管理**
```bash
openclaw pairing list                    # 查看所有配对请求
openclaw pairing approve telegram <码>   # 批准
openclaw pairing reject telegram <码>    # 拒绝
```

## 使用

```shell
# 在终端进入交互
openclaw tui
# 第一次对话会询问你是否需要给他人设等，可以花点时间设置好
```

### 会话管理

子会话用于异步执行任务，有两种模式：
- **run 模式** - 一次性任务，执行完自动结束
- **session 模式** - 持久会话，可多次交互

**创建子会话**

在对话中直接说（但是好像终端不可以，只能在飞书）：

```shell
"创建一个子会话来分析这个日志"
"用子会话帮我处理这个数据"
"创建持久子会话来监控服务器状态"
```

**切换子会话**

```shell
/session
/sessions
```

**重要特性**
- 异步执行，不阻塞主会话
- 任务完成自动通知
- 独立上下文和历史

### agent 管理

**创建 agent**

跟创建会话一样

**切换 agent**

```shell
/agent
/agents
```


## 基本命令

```bash
openclaw tui               # 启动对话
openclaw status            # 查看状态
openclaw configure         # 修改配置
openclaw sessions          # 查看所有会话
openclaw --help            # 帮助
```

**会话中的命令**
```bash
/status      # 查看会话状态（token 使用、时间、费用）
/reasoning   # 切换推理模式
/new         # 清空历史
Ctrl+C       # 退出
```

## 目录结构

```
~/.openclaw/
├── openclaw.json          # 主配置文件
├── openclaw.json.bak*     # 配置文件备份（自动生成）
├── workspace/             # AI 工作区
│   ├── SOUL.md           # AI 的性格、行为准则
│   ├── IDENTITY.md       # AI 的名字、emoji
│   ├── USER.md           # 你的信息、偏好
│   ├── AGENTS.md         # 工作流程规则（系统文件）
│   ├── MEMORY.md         # 长期记忆（仅主会话加载）
│   ├── HEARTBEAT.md      # 定期检查任务
│   ├── TOOLS.md          # 本地工具配置
│   ├── BOOTSTRAP.md      # 首次运行引导（自动删除）
│   ├── WORKFLOW_AUTO.md  # 自动化工作流配置
│   └── memory/
│       ├── YYYY-MM-DD.md         # 每日记录
│       ├── heartbeat-state.json  # 心跳状态
│       └── main.sqlite           # 记忆数据库
├── agents/                # Agent 实例数据
│   └── main/             # 主 Agent 数据
├── credentials/           # 凭证和配对信息
│   ├── telegram-pairing.json
│   └── telegram-default-allowFrom.json
├── devices/               # 配对设备信息
│   ├── paired.json
│   └── pending.json
├── identity/              # 设备身份信息
│   ├── device.json
│   └── device-auth.json
├── telegram/              # Telegram 频道数据
├── subagents/             # 子会话运行记录
│   └── runs.json
├── cron/                  # 定时任务配置
│   └── jobs.json
├── memory/                # 记忆系统数据库
│   └── main.sqlite
├── logs/                  # 日志文件
│   ├── gateway.log
│   ├── gateway.err.log
│   └── config-audit.jsonl
├── completions/           # Shell 自动补全脚本
│   ├── openclaw.bash
│   ├── openclaw.zsh
│   └── openclaw.fish
├── canvas/                # Canvas 界面资源
├── delivery-queue/        # 消息投递队列
└── exec-approvals.json    # 命令执行审批记录
```

首次启动会提示配置端口、密码、HTTPS。默认访问地址：`http://localhost:3000`

## 记忆系统

**自动记忆**
AI 自动将重要信息写入 `memory/YYYY-MM-DD.md`

**长期记忆**
`MEMORY.md` 存储长期重要信息（仅主会话加载，不在群聊中加载）

**让 AI 记住**
```
"记住：我喜欢用 Vim 编辑器"
"把这个记下来：项目截止日期是 3 月 15 日"
```

**手动编辑**
```bash
vim ~/.openclaw/workspace/MEMORY.md
cat ~/.openclaw/workspace/memory/$(date +%Y-%m-%d).md
```

## 常见问题

**安装和配置**
- 安装后找不到命令？重启终端或运行 `source ~/.zshrc`
- 如何更新？重新运行安装脚本：`curl -fsSL https://openclaw.ai/install.sh | bash`
- 如何卸载？删除配置和可执行文件：`rm -rf ~/.openclaw && rm /usr/local/bin/openclaw`

**模型相关**
- 上下文被限制在 4096？编辑 `~/.openclaw/openclaw.json`，修改 `contextWindow` 值
- 支持哪些模型？所有兼容 OpenAI API 的模型（GPT、Claude、DeepSeek、本地模型等）

**使用相关**
- 如何让 AI 记住信息？直接说"记住这个"，或手动编辑 `MEMORY.md`
- 如何重置 AI？`mv ~/.openclaw/workspace ~/.openclaw/workspace.backup && openclaw chat`
- 如何修改 AI 风格？编辑 `~/.openclaw/workspace/SOUL.md`
- 多设备同步？用 Git 管理 `~/.openclaw/workspace` 目录

**Session 和 Agent**
- Session 和 Agent 有什么区别？Agent 是 AI 实例，Session 是对话实例
- 如何创建子会话？在对话中说"创建一个子会话来..."
- 子会话的两种模式？`run` 模式执行完自动结束，`session` 模式可持续交互
- 为什么无法"进入"子会话？子会话是异步后台任务，不是交互式终端
- 如何与子会话交互？对于持久子会话，说"向子会话 xxx 发送消息：..."
- 如何终止子会话？说"终止子会话 xxx"
- 如何创建自定义 Agent？目前不支持通过配置文件创建，可使用内置 Sub-Agent

**网关相关**
- 网关启动失败？检查端口占用 `lsof -i :3000`，或更换端口
- 忘记密码？编辑 `~/.openclaw/openclaw.json`，删除 `gateway.password` 字段

## 进阶技巧

**自定义 AI 性格**
编辑 `~/.openclaw/workspace/SOUL.md`：
```markdown
# SOUL.md

## 核心原则
- 代码优先：能写代码就不要只给建议
- 简洁高效：避免废话，直接解决问题
- 安全第一：涉及敏感操作时必须确认

## 风格
- 技术讨论：专业、精确
- 日常对话：轻松、友好
```

**工作流自动化**
编辑 `HEARTBEAT.md` 实现定期检查：
```markdown
# 每天早上 9:00 检查
- 查看未读邮件（重要的）
- 检查今天的日历事件
- 汇报天气情况
```

## 资源链接

- 官方文档: https://docs.openclaw.ai/
- 中文文档: https://openclawcn.cn/docs.html
- GitHub: https://github.com/openclaw/openclaw
- Discord 社区: https://discord.com/invite/clawd
- 技能市场: https://clawhub.com
- 本地文档: `/opt/homebrew/lib/node_modules/openclaw/docs`

---

最后更新: 2026-03-05
