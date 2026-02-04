

claudecode 官网：[https://code.claude.com/docs/zh-CN/overview](https://code.claude.com/docs/zh-CN/overview)

# 环境配置

**安装**

```shell
# 报错大概率是 unavailable region 网络重定向了
curl -fsSL https://claude.ai/install.sh | bash
```

**更改安装版本（若需）**

```shell
claude install 2.1.17
```

**禁用自动更新（若需）**

```shell
# 在 ~/.claude.json 的 env 字段中添加如下设置：
{
  "env": {
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

**使用外部 API（若需）**

```shell
# 添加如下环境变量
export ANTHROPIC_BASE_URL="xxx"
export ANTHROPIC_AUTH_TOKEN="xxx"
export ANTHROPIC_API_KEY="xxx"
```