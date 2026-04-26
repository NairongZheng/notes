- [安装](#安装)
- [使用](#使用)
  - [基础命令](#基础命令)

github 链接：[https://github.com/nousresearch/hermes-agent](https://github.com/nousresearch/hermes-agent)

# 安装

```shell
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

# 使用

## 基础命令

```shell
📁 All your files are in ~/.hermes/:

   Settings:  /Users/zhengnairong/.hermes/config.yaml
   API Keys:  /Users/zhengnairong/.hermes/.env
   Data:      /Users/zhengnairong/.hermes/cron/, sessions/, logs/

────────────────────────────────────────────────────────────

📝 To edit your configuration:

   hermes setup          Re-run the full wizard
   hermes setup model    Change model/provider
   hermes setup terminal Change terminal backend
   hermes setup gateway  Configure messaging
   hermes setup tools    Configure tool providers

   hermes config         View current settings
   hermes config edit    Open config in your editor
   hermes config set <key> <value>
                          Set a specific value

   Or edit the files directly:
   nano /Users/zhengnairong/.hermes/config.yaml
   nano /Users/zhengnairong/.hermes/.env

────────────────────────────────────────────────────────────

🚀 Ready to go!

   hermes              Start chatting
   hermes gateway      Start messaging gateway
   hermes doctor       Check for issues
```