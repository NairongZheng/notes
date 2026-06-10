- [安装](#安装)
- [初始化设置](#初始化设置)
  - [一键设置](#一键设置)
  - [其他设置](#其他设置)
    - [channels 设置](#channels-设置)
    - [模型设置](#模型设置)
- [使用](#使用)


# 安装

```shell
pip install nanobot-ai
```

**命令**

```shell
(dev) ➜  /Users/zhengnairong/code/agent_learning %  git:(main) ✗ nanobot --help

 Usage: nanobot [OPTIONS] COMMAND [ARGS]...

 🐈 nanobot - Personal AI Assistant

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --version             -v                                                                                                                                                                                 │
│ --install-completion            Install completion for the current shell.                                                                                                                                │
│ --show-completion               Show completion for the current shell, to copy it or customize the installation.                                                                                         │
│ --help                          Show this message and exit.                                                                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ onboard    Initialize nanobot configuration and workspace.                                                                                                                                               │
│ gateway    Start the nanobot gateway.                                                                                                                                                                    │
│ agent      Interact with the agent directly.                                                                                                                                                             │
│ status     Show nanobot status.                                                                                                                                                                          │
│ channels   Manage channels                                                                                                                                                                               │
│ plugins    Manage channel plugins                                                                                                                                                                        │
│ provider   Manage providers                                                                                                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

# 初始化设置

## 一键设置

```shell
nanobot onboard
```

**日志**

```shell
(dev) ➜  /Users/zhengnairong/code/agent_learning %  git:(main) ✗ nanobot onboard
✓ Created config at /Users/zhengnairong/.nanobot/config.json
Config template now uses `maxTokens` + `contextWindowTokens`; `memoryWindow` is no longer a runtime setting.
2026-03-20 17:54:34.345 | DEBUG    | nanobot.channels.registry:discover_all:64 - Skipping built-in channel 'matrix': Matrix dependencies not installed. Run: pip install nanobot-ai[matrix]
  Created HEARTBEAT.md
  Created USER.md
  Created SOUL.md
  Created AGENTS.md
  Created TOOLS.md
  Created memory/MEMORY.md
  Created memory/HISTORY.md

🐈 nanobot is ready!

Next steps:
  1. Add your API key to ~/.nanobot/config.json
     Get one at: https://openrouter.ai/keys
  2. Chat: nanobot agent -m "Hello!"

Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps
```

## 其他设置

**剩下的大部分配置都可以在 `~/.nanobot/config.json` 中设置**

### channels 设置

这边还是以 telegram 举例

1. 找 [@BotFather](https://t.me/BotFather) 创建机器人
2. 发送 `/newbot` 并按提示操作
3. 获取 Bot Token
4. 找 [@userinfobot](https://t.me/userinfobot) 随便发个消息获得自己的 user-id
5. 在 `~/.nanobot/config.json` 的 `channels` 的 `telegram` 中填入正确：
   1. enabled：`true`
   2. token：`"bot token"`
   3. allowFrom：`["user-id"]`

### 模型设置

在 `~/.nanobot/config.json` 中修改以下：
1. `providers` 中选择正确的模型，填入正确的 `apiKey` 与 `apiBase`
2. `agents` 中选择修改的 agent 的 `model` 和 `provider`
3. `maxTokens` 跟 `contextWindowTokens` 也可以修改一下


# 使用

1. 经过上面的配置之后，启动网关（`nanobot gateway`）后，telegram 就可以使用了
2. `nanobot agent` 可以直接进入交互模式聊天，可以看看 agent 这个命令的参数：

```shell
(dev) ➜  /Users/zhengnairong/code %  nanobot agent --help

 Usage: nanobot agent [OPTIONS]

 Interact with the agent directly.

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --message    -m                   TEXT  Message to send to the agent                                                                                                                                     │
│ --session    -s                   TEXT  Session ID [default: cli:direct]                                                                                                                                 │
│ --workspace  -w                   TEXT  Workspace directory                                                                                                                                              │
│ --config     -c                   TEXT  Config file path                                                                                                                                                 │
│ --markdown       --no-markdown          Render assistant output as Markdown [default: markdown]                                                                                                          │
│ --logs           --no-logs              Show nanobot runtime logs during chat [default: no-logs]                                                                                                         │
│ --help                                  Show this message and exit.                                                                                                                                      │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```