# OpenClaw 完全指南

> 一个让 AI 助手真正融入你工作流的开源框架

**官网**: https://openclaw.ai/ | **文档**: https://docs.openclaw.ai/ | **社区**: [Discord](https://discord.com/invite/clawd)

---

## 目录

- [快速开始](#快速开始)
- [核心概念](#核心概念)
- [目录结构](#目录结构)
- [配置指南](#配置指南)
- [使用教程](#使用教程)
- [常见问题](#常见问题)

---

## 快速开始

### 安装

```bash
# macOS / Linux 一键安装
curl -fsSL https://openclaw.ai/install.sh | bash

# 重启终端或运行
source ~/.zshrc  # 或 source ~/.bashrc

# 验证安装
openclaw --version
openclaw status
```

### 首次运行

```bash
openclaw chat
```

首次运行时，AI 会引导你完成初始化：
1. 设置 AI 的名字（比如"小白"）
2. 定义 AI 的性格和风格
3. 填写你的基本信息
4. 选择时区和偏好设置

完成后，`BOOTSTRAP.md` 会自动删除，你的 AI 助手就准备好了！

---

## 核心概念

### Session（会话）

Session 是你与 AI 的一次对话实例。每个 session 都有独立的：
- 对话历史
- 上下文状态
- Token 使用统计

**Session 类型：**

| 类型 | 说明 | 使用场景 |
|------|------|----------|
| **主会话** | 直接与 AI 对话，可访问所有记忆文件 | 日常使用、个人助手 |
| **子会话** | 通过 `sessions_spawn` 创建的独立会话 | 并行任务、隔离上下文 |
| **线程会话** | 绑定到消息线程的持久会话 | Discord/Telegram 群聊 |

**Session 管理：**
```bash
# 查看所有会话
openclaw sessions list

# 向其他会话发送消息
openclaw sessions send <session-key> "消息内容"

# 查看会话历史
openclaw sessions history <session-key>
```

### Agent（代理）

Agent 是具有特定能力和配置的 AI 实例。一个 Agent 可以运行多个 Session。

**Agent 类型：**

| 类型 | 说明 | 使用场景 |
|------|------|----------|
| **主 Agent** | 你的个人 AI 助手 | 日常对话、任务处理 |
| **Sub-Agent** | 专门处理特定任务的 Agent | 代码审查、数据分析 |
| **Coding Agent** | 专门用于编程的 Agent（Codex/AWS Code） | 创建项目、重构代码 |

**关系图：**
```
Agent (AI 实例)
  ├── Session 1 (主会话)
  ├── Session 2 (子会话 - 处理任务 A)
  └── Session 3 (子会话 - 处理任务 B)
```

**创建子会话：**
```bash
# 在对话中说
"创建一个子会话来分析这个日志文件"

# 或使用命令
openclaw sessions spawn --task "分析日志" --mode session
```

**Sub-Agent 使用：**
- `context-gatherer` - 探索代码库，识别相关文件
- `general-task-execution` - 执行独立的子任务
- `subagent-creator` - 创建新的专用 Agent

### Skill（技能）

Skill 是预定义的专业能力模块，为 Agent 提供特定功能。

**系统技能位置：** `/opt/homebrew/lib/node_modules/openclaw/skills/`  
**自定义技能位置：** `~/.openclaw/skills/`

**常用技能：**
- `coding-agent` - 委托编码任务
- `weather` - 获取天气信息
- `tmux` - 远程控制 tmux 会话
- `skill-creator` - 创建自定义技能

---

## 目录结构

### 主配置目录

```
~/.openclaw/
├── openclaw.json          # 主配置（模型、网关等）
├── workspace/             # AI 工作区（记忆、身份）
├── settings/
│   └── mcp.json          # MCP 服务器配置
├── skills/                # 自定义技能
└── logs/                  # 日志文件
```

### 工作区文件

`~/.openclaw/workspace/` 是 AI 的"家"：

**核心身份文件：**
- `SOUL.md` - AI 的性格、行为准则
- `IDENTITY.md` - AI 的名字、emoji
- `USER.md` - 你的信息、偏好
- `AGENTS.md` - 工作流程规则（系统文件）

**记忆系统：**
- `MEMORY.md` - 长期记忆（仅主会话）
- `memory/YYYY-MM-DD.md` - 每日记录
- `memory/heartbeat-state.json` - 心跳状态

**功能配置：**
- `HEARTBEAT.md` - 定期检查任务
- `TOOLS.md` - 本地工具配置
- `BOOTSTRAP.md` - 首次运行引导（自动删除）

---

## 配置指南

### 模型配置

**方法一：交互式配置（推荐）**

```bash
openclaw configure
```

按照以下步骤操作：

1. 选择 `Model` → `Custom Provider`
2. 输入 API Base URL
   - 示例：`https://api.anthropic.com/v1`
   - 或：`https://api.deepseek.com/v1`
   - 或：`https://api.openai.com/v1`
3. 输入 API Key
   - 示例：`sk-ant-xxxxxxxxxxxxx`
4. 选择类型
   - `OpenAI` - 适用于 OpenAI、DeepSeek 等兼容 API
   - `Anthropic` - 适用于 Claude 系列
   - `Other` - 其他自定义 API
5. 填写 Model ID
   - Claude 示例：`claude-sonnet-4-5-20250929`
   - DeepSeek 示例：`deepseek-chat`
   - GPT 示例：`gpt-4-turbo`

**方法二：手动编辑**

编辑 `~/.openclaw/openclaw.json`：

```json
{
  "models": {
    "providers": {
      "custom-api-claude": {
        "type": "anthropic",
        "baseURL": "https://api.anthropic.com/v1",
        "apiKey": "sk-ant-xxxxxxxxxxxxx",
        "models": {
          "claude-sonnet-4-5-20250929": {
            "id": "claude-sonnet-4-5-20250929",
            "contextWindow": 200000,
            "maxTokens": 8192
          }
        }
      },
      "custom-api-deepseek": {
        "type": "openai",
        "baseURL": "https://api.deepseek.com/v1",
        "apiKey": "sk-xxxxxxxxxxxxx",
        "models": {
          "deepseek-chat": {
            "id": "deepseek-chat",
            "contextWindow": 64000,
            "maxTokens": 4096
          }
        }
      }
    }
  }
}
```

**⚠️ 重要：修改上下文窗口**

OpenClaw 对未知模型默认限制为 4096 tokens，需手动修改 `contextWindow`。

### 频道配置

频道（Channel）让 AI 通过消息平台与你交互（WhatsApp、Telegram、Discord 等）。

**支持的频道：**
- WhatsApp
- Telegram
- Discord
- Signal
- iMessage
- Slack
- IRC
- Google Chat

**配置方法：**

```bash
openclaw configure
# 选择 Channels → 选择要配置的频道
```

#### WhatsApp 配置

1. 运行 `openclaw configure` → `Channels` → `WhatsApp`
2. 选择 `Link Account`
3. 扫描显示的二维码（用 WhatsApp 扫码）
4. 连接成功后，可以通过 WhatsApp 与 AI 对话

**注意：** 需要保持 OpenClaw 运行才能接收消息。

#### Telegram 配置

1. 在 Telegram 中找 [@BotFather](https://t.me/BotFather)
2. 发送 `/newbot` 创建新机器人
3. 按提示设置机器人名称和用户名
4. 获取 Bot Token（格式：`123456789:ABCdefGHIjklMNOpqrsTUVwxyz`）
5. 运行 `openclaw configure` → `Channels` → `Telegram`
6. 输入 Bot Token
7. 在 Telegram 中搜索你的机器人并发送 `/start`

**首次连接配对：**

第一次向机器人发送消息时，会收到配对请求：

```
OpenClaw: access not configured.

Your Telegram user id: xxx

Pairing code: xxx

Ask the bot owner to approve with:
openclaw pairing approve telegram xxx
```

在终端运行批准命令：

```bash
openclaw pairing approve telegram xxx
```

批准后，回到 Telegram 重新发送消息即可正常对话。

**配对管理命令：**

```bash
# 查看所有配对请求
openclaw pairing list

# 批准配对
openclaw pairing approve telegram <配对码>

# 拒绝配对
openclaw pairing reject telegram <配对码>
```

**说明：**
- 配对机制防止未授权用户访问你的 AI
- 每个新用户首次连接都需要批准
- 批准后该用户 ID 会被记住，以后不需要再批准

#### Discord 配置

1. 访问 [Discord Developer Portal](https://discord.com/developers/applications)
2. 创建新应用（New Application）
3. 进入 Bot 页面，点击 `Add Bot`
4. 复制 Bot Token
5. 在 OAuth2 → URL Generator 中：
   - 勾选 `bot` scope
   - 勾选需要的权限（至少：Send Messages、Read Message History）
   - 复制生成的 URL，在浏览器中打开并邀请机器人到服务器
6. 运行 `openclaw configure` → `Channels` → `Discord`
7. 输入 Bot Token

**手动配置（高级）：**

编辑 `~/.openclaw/openclaw.json`：

```json
{
  "channels": {
    "telegram": {
      "enabled": true,
      "botToken": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
    },
    "discord": {
      "enabled": true,
      "botToken": "your-discord-bot-token",
      "guildId": "your-server-id"
    },
    "whatsapp": {
      "enabled": true
    }
  }
}
```

**使用频道：**

配置完成后，AI 会自动响应来自这些频道的消息。你可以：
- 在 WhatsApp/Telegram/Discord 中直接与 AI 对话
- 在群组中 @机器人 来触发响应
- 使用 `/status`、`/help` 等命令

### MCP 配置

MCP (Model Context Protocol) 让 AI 访问外部工具和数据源。

**配置文件：** `~/.openclaw/settings/mcp.json`

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxx"
      }
    }
  }
}
```

**常用 MCP 服务器：**
- `@modelcontextprotocol/server-filesystem` - 文件系统
- `@modelcontextprotocol/server-github` - GitHub 集成
- `@modelcontextprotocol/server-postgres` - PostgreSQL
- `@modelcontextprotocol/server-brave-search` - Brave 搜索

### 网关配置

网关提供网页端访问。

```bash
# 启动网关
openclaw gateway start

# 管理命令
openclaw gateway status   # 查看状态
openclaw gateway stop     # 停止
openclaw gateway restart  # 重启
```

首次启动会提示配置端口、密码、HTTPS。访问地址通常是 `http://localhost:3000`。

### 技能配置

**创建自定义技能：**

方式一：让 AI 帮你创建
```
"帮我创建一个技能，用于..."
```

方式二：手动创建
```
~/.openclaw/skills/my-skill/
├── SKILL.md       # 技能说明
├── script.sh      # 可执行脚本（可选）
└── assets/        # 资源文件（可选）
```

---

## 使用教程

### 命令行使用

```bash
# 启动会话
openclaw chat

# 常用命令
openclaw status              # 查看状态
openclaw configure           # 修改配置
openclaw --help              # 帮助

# 会话中的命令
/status      # 查看会话状态
/reasoning   # 切换推理模式
/clear       # 清空历史
Ctrl+C       # 退出
```

### 网页端使用

1. 启动网关：`openclaw gateway start`
2. 浏览器访问提示的地址
3. 输入密码（如果设置了）
4. 开始对话

**网页端优势：** 更友好的界面、多标签页、文件上传、代码高亮

### 记忆系统

**自动记忆：**
AI 自动将重要信息写入 `memory/YYYY-MM-DD.md`

**长期记忆：**
`MEMORY.md` 存储长期重要信息（仅主会话加载）

**让 AI 记住：**
```
"记住：我喜欢用 Vim 编辑器"
"把这个记下来：项目截止日期是 3 月 15 日"
```

**手动编辑：**
```bash
# 编辑长期记忆
vim ~/.openclaw/workspace/MEMORY.md

# 查看今天的记录
cat ~/.openclaw/workspace/memory/$(date +%Y-%m-%d).md
```

### Session 和 Agent 实战

**场景 1：并行处理多个任务**

```bash
# 主会话中
"创建一个子会话来分析日志文件"
"再创建一个子会话来重构代码"

# 查看所有会话
openclaw sessions

# 向子会话发送指令
openclaw sessions send <session-key> "继续分析"
```

**场景 2：使用 Coding Agent**

```bash
# 在对话中
"用 coding-agent 创建一个 React 项目"

# 或直接委托
"帮我重构这个代码库，使用 coding-agent"
```

**场景 3：探索陌生代码库**

```bash
# 使用 context-gatherer sub-agent
"用 context-gatherer 帮我找出登录相关的文件"
```

---

## 常见问题

### 安装和配置

**Q: 安装后找不到命令？**  
A: 重启终端或 `source ~/.zshrc`

**Q: 如何更新？**  
A: 重新运行安装脚本

**Q: 如何卸载？**  
A: `rm -rf ~/.openclaw` 和删除可执行文件

### 模型相关

**Q: 上下文被限制在 4096？**  
A: 编辑 `openclaw.json`，修改 `contextWindow` 值

**Q: 支持哪些模型？**  
A: 所有兼容 OpenAI API 的模型（GPT、Claude、DeepSeek、本地模型等）

**Q: 如何使用本地模型（Ollama）？**  
A:
```bash
ollama serve
openclaw configure
# Base URL: http://localhost:11434/v1
# Model ID: llama2
```

### 使用相关

**Q: 如何让 AI 记住信息？**  
A: 直接说"记住这个"，或编辑 `MEMORY.md`

**Q: 如何重置 AI？**  
A:
```bash
mv ~/.openclaw/workspace ~/.openclaw/workspace.backup
openclaw chat
```

**Q: 如何修改 AI 风格？**  
A: 编辑 `~/.openclaw/workspace/SOUL.md`

**Q: 多设备同步？**  
A: 用 Git 管理 `~/.openclaw/workspace` 目录

### Session 和 Agent

**Q: Session 和 Agent 有什么区别？**  
A: Agent 是 AI 实例，Session 是对话实例。一个 Agent 可以运行多个 Session。

**Q: 什么时候用子会话？**  
A: 需要并行处理任务、隔离上下文、或委托给专门的 Agent 时。

**Q: 如何查看所有会话？**  
A: `openclaw sessions list` 或在对话中说"列出所有会话"

**Q: 子会话会共享记忆吗？**  
A: 不会。子会话有独立的上下文，但可以通过 `sessions_send` 通信。

### 网关相关

**Q: 网关启动失败？**  
A: 检查端口占用 `lsof -i :3000`，或更换端口

**Q: 忘记密码？**  
A: 编辑 `openclaw.json`，删除 `gateway.password`

### 高级问题

**Q: 如何定期检查邮件/日历？**  
A: 编辑 `HEARTBEAT.md`，添加检查任务

**Q: 如何集成 Discord/Telegram？**  
A: 参考官方文档的 Messaging Integration 章节

**Q: 如何创建自定义 Sub-Agent？**  
A: 使用 `subagent-creator` 技能或在对话中说"创建一个专门用于...的 agent"

---

## 进阶技巧

### 自定义 AI 性格

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

### 工作流自动化

编辑 `HEARTBEAT.md` 实现定期检查：

```markdown
# 每天早上 9:00 检查
- 查看未读邮件（重要的）
- 检查今天的日历事件
- 汇报天气情况
```

### 团队协作

**共享技能库：**
```bash
git clone https://github.com/your-team/openclaw-skills ~/.openclaw/skills/team
```

**统一配置：**
用 Git 管理配置文件（注意排除敏感信息）

---

## 资源链接

- **官方文档**: https://docs.openclaw.ai/
- **GitHub**: https://github.com/openclaw/openclaw
- **Discord 社区**: https://discord.com/invite/clawd
- **技能市场**: https://clawhub.com
- **本地文档**: `/opt/homebrew/lib/node_modules/openclaw/docs`

---

**最后更新**: 2026-03-04

_这份文档会随着使用不断完善。有问题随时问 AI！_
