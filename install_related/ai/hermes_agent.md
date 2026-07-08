- [安装](#安装)
- [使用](#使用)
  - [基础命令](#基础命令)
  - [一些配置](#一些配置)
- [GitLab MR 自动合并与飞书同步](#gitlab-mr-自动合并与飞书同步)

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

───────────────────────────────────────────────────────────

🚀 Ready to go!

   hermes              Start chatting
   hermes gateway      Start messaging gateway
   hermes doctor       Check for issues
```

## 一些配置

```shell
# 设置是保存 session.json 还是用 db 存储（默认 false）
hermes config set sessions.write_json_snapshots false
```

# GitLab MR 自动合并与飞书同步

**1. 工作流程**

```text
GitLab MR open/update
  -> 检查 Pipeline、Job、文件路径和文件大小
  -> 全部通过后自动合并
  -> 飞书群通知合并结果；未通过时通知阻塞原因供人工 Review
  -> 收到 MR merge 事件
  -> 按正文识别数据集质检 Markdown（文件名不限）
  -> 对比飞书数据合成指导
  -> 有新的通用建议时才更新飞书
```

自动合并提交使用标识：

```text
[Hermes Auto Merge] MR !<iid>: <MR title>
```

**2. Hermes 配置**

`hermes config edit` 加入：

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      host: 0.0.0.0
      port: 8644
      secret: "用 openssl rand -hex 32 生成"

command_allowlist:
  - /Users/zhengnairong/.hermes/bin/hermes-gitlab-review *
  - /Users/zhengnairong/.hermes/bin/hermes-gitlab-review-by-sha *
  - /Users/zhengnairong/.hermes/bin/hermes-feishu-sync *
  - /Users/zhengnairong/.hermes/bin/hermes-gitlab-review-notify *
  - /Users/zhengnairong/.hermes/bin/hermes-gitlab-review-by-sha-notify *
  - /Users/zhengnairong/.hermes/bin/hermes-feishu-sync-notify *
```

开启 Webhook 终端工具，并关闭无人值守任务的交互提问：

```shell
hermes tools enable terminal --platform webhook
hermes tools disable clarify --platform webhook
hermes gateway restart
```

GitLab CLI 登录：

```shell
brew install glab
glab auth login
glab auth status
```

**3. 创建动态 Webhook**

长 Prompt 保存在：

```text
~/.hermes/prompts/data_mllm_agent_webhook.txt
~/.hermes/prompts/data_mllm_agent_feishu_sync.txt
~/.hermes/prompts/data_mllm_agent_pipeline.txt
```

`data_mllm_agent_webhook.txt` 负责分流：

```text
open / reopen / update -> 审查、符合规则时自动合并并通知飞书群
merge                  -> 按正文识别质检 Markdown 并同步飞书
```

`--prompt "$(cat ...)"` 会在创建路由时读取 Prompt 文件内容，因此一个 Webhook 可以同时完成自动合并和合并后飞书同步。

创建路由：

```shell
GITLAB_SECRET="$(openssl rand -hex 32)"

hermes webhook subscribe data_mllm_agent \
  --events "Merge Request Hook" \
  --secret "$GITLAB_SECRET" \
  --deliver log \
  --description "data_mllm_agent 自动合并与飞书质检同步" \
  --prompt "$(cat ~/.hermes/prompts/data_mllm_agent_webhook.txt)"

echo "$GITLAB_SECRET"
hermes webhook ls
```

创建 Pipeline 成功后的二次审查路由：

```shell
PIPELINE_SECRET="$(openssl rand -hex 32)"

hermes webhook subscribe data_mllm_agent_pipeline \
  --events "Pipeline Hook" \
  --secret "$PIPELINE_SECRET" \
  --deliver log \
  --description "data_mllm_agent Pipeline 成功后自动合并" \
  --prompt "$(cat ~/.hermes/prompts/data_mllm_agent_pipeline.txt)"

echo "$PIPELINE_SECRET"
```

修改 Prompt 文件后，需重新执行 `hermes webhook subscribe` 更新路由。

**4. GitLab 配置**

```text
URL: http://<本机局域网 IP>:8644/webhooks/data_mllm_agent
Secret token: GITLAB_SECRET
Trigger: Merge request events
```

再新增一个 Pipeline Webhook：

```text
URL: http://<本机局域网 IP>:8644/webhooks/data_mllm_agent_pipeline
Secret token: PIPELINE_SECRET
Trigger: Pipeline events
```

Pipeline 成功后，Hermes 使用 Pipeline SHA 精确定位当前 opened MR，再运行一次完整审查和自动合并，并将结果通知飞书群。

**5. 当前审查规则**

- 仅允许 `agent_data/data_mllm_agent` 合并到 `main`。
- Pipeline 和所有 Job 必须是 `success`。
- 只允许 `data_selfmade/`、`data_opensource/`、`data_examples/`。
- 单文件上限为 10 MiB。
- Draft、冲突、未解决讨论或仓库外层配置变更均不合并。

策略文件：

```text
~/.hermes/config/gitlab_review_policy.json
~/.hermes/config/feishu_sync_policy.json
```

**6. 飞书配置**

飞书应用需要文档权限，以及机器人能力、`im:message:send_as_bot` 和 `im:chat:readonly` 权限。

```shell
export FEISHU_REVIEW_CHAT_ID="oc_xxx"
```

```text
文档：stage0-数据合成指导（持续更新）
document_id: xxx
```

同步会扫描数据集目录内的 `.md/.markdown`，根据“质检报告、数据质量、异常率、重复率、QC report”以及 LLMChecker 检查项等正文标志识别报告，不要求固定文件名。超长报告会保留开头和结尾，普通辅助 Markdown 只读取少量内容。同步只修改“Hermes 自动同步的数据质检建议”章节，并用 `project + MR IID + merge SHA` 避免重复写入。

飞书正文使用易读的中文描述问题和建议，不直接展示质检字段名、英文缩写或内部规则名称。

通知按三个独立阶段去重：`received` 表示收到 MR，`review_result` 表示自动审查结果，`sync_result` 表示合并后的文档同步结果。正常自动合并通常发送“收到 + 同步完成”两条；若审查失败后又人工合并，则会继续补发文档同步结果。MR Hook 与 Pipeline Hook 不会重复发送同一阶段。

自动章节模板：

```text
数据合成风险与建议
├─ 当前可能存在的问题（最多 5 条）
├─ 后续更新或合成时需要注意（最多 6 条）
└─ 更新记录（时间、MR、数据集）
```

**7. 测试和日志**

```shell
curl http://127.0.0.1:8644/health
tail -f ~/.hermes/logs/gateway.log

# 只审查，不合并
~/.hermes/bin/hermes-gitlab-review agent_data/data_mllm_agent 36

# 审查、符合规则时自动合并并通知飞书群
~/.hermes/bin/hermes-gitlab-review-notify agent_data/data_mllm_agent 36 --merge

# 飞书对比，不写入
~/.hermes/bin/hermes-feishu-sync agent_data/data_mllm_agent 36 --dry-run

# 真实同步飞书
~/.hermes/bin/hermes-feishu-sync agent_data/data_mllm_agent 36
```
