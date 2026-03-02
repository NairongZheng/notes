

官网：[https://openclaw.ai/](https://openclaw.ai/)

文档：[https://docs.openclaw.ai/](https://docs.openclaw.ai/)

# 安装

**一键安装**

```shell
curl -fsSL https://openclaw.ai/install.sh | bash
```

**查看状态**

```shell
openclaw status
```

# 配置

使用以下命令修改配置：

```shell
openclaw configure
```

**配置网关**

配置网关后可以在网页端使用 openclaw

**配置自定义模型**

```shell
# 1. 进入配置
openclaw configure
# 2. 选择 Model -> Custom Provider
# 3. 输入 api base url 跟 key
# 4. 选择类型 OpenAI/Anthropic/Other
# 5. 填 Model ID，如：claude-sonnet-4-5-20250929
```

使用自定义模型之后，需要到 `~/.openclaw/openclaw.json` 里面的 `models` 的对应的 `providers` 修改 `contextWindow`

因为 openclaw 在处理不在内部数据库中的模型时，出于安全考虑，会把上下文限制在 4096