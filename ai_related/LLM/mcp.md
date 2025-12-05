
- [简介](#简介)
- [mcp server 格式说明](#mcp-server-格式说明)
  - [npx 系列 —— npm 上的 MCP Server](#npx-系列--npm-上的-mcp-server)
  - [uvx / uv run 系列 —— Python MCP Server](#uvx--uv-run-系列--python-mcp-server)
  - [mcp-remote 系列 —— 远程托管的 MCP](#mcp-remote-系列--远程托管的-mcp)
  - [smithery 系列 —— MCP 工具集市（最常见）](#smithery-系列--mcp-工具集市最常见)
  - [SSE-only server（无需安装，直接 URL）](#sse-only-server无需安装直接-url)

# 简介

MCP(Model Context Protocol) 统一标准，由 Anthropic 牵头制定。

它允许不同的工具 (Tools) 作为独立的 MCP Server 运行，通过标准协议向 LLM 暴露工具能力。

[mcp 官方链接](https://modelcontextprotocol.io/docs/getting-started/intro)

在这之前要提前安装几个需要的包：
1. [安装 nvm](../../dev_related/op_and_cmd.md) 来使用 `npm` 与 `mpx`
2. [安装 uv](../../install_related/uv_usage.md) 来使用 `uv run` 与 `uvx`

几个可以查看或者搜索 mcp server 的网址：
1. [Smithery](https://smithery.ai/skills)
2. [Model Context Protocol 官方仓库列表](https://github.com/modelcontextprotocol/servers)
3. [npm](https://www.npmjs.com/)
4. [PyPI](https://pypi.org/)

# mcp server 格式说明

mcp server 有不同类型的包，以下分别说明。

## npx 系列 —— npm 上的 MCP Server

```shell
"mcp-deepwiki": {
    "command": "npx",
    "args": [
        "-y",
        "mcp-deepwiki@latest"
    ]
}
```

MCP 是 server 模式运行的，调用方式通常是：

```shell
npx -y mcp-deepwiki@latest
```

有些包运行完是没有 log 的，只能接受 client 连接。

可以使用 vscode 的插件 mcp inspector 来查看具体这个 mcp 有哪些 tools

![](../../images/2025/20251205_1_mcp_npx.png)

## uvx / uv run 系列 —— Python MCP Server

```shell
"weibo": {
    "command": "uvx",
    "args": [
        "mcp-server-weibo"
    ]
}
```

python 的 mcp server 跟 npm 的有点不一样。可以使用以下命令查看可执行文件：

```shell
# (dev) zhengnairong@xxxxxx Downloads % uvx --with mcp-server-weibo python - << 'EOF'
# import sysconfig, os
# scripts = sysconfig.get_path("scripts")
# print("Scripts path:", scripts)
# print("Files:", os.listdir(scripts))
# EOF
# Scripts path: /Users/zhengnairong/.cache/uv/archive-v0/BeseviwOD5TzY_kiJTEPJ/bin
# Files: ['activate.bat', 'rst2html', 'activate.ps1', 'dotenv', 'python3', 'typer', 'rst2xml', 'docutils', 'python', 'rst2man', 'activate.fish', 'rst2odt', 'fastmcp', 'websockets', 'python3.11', 'pydoc.bat', 'mcp', 'activate_this.py', 'cyclopts', 'httpx', 'jsonschema', 'markdown-it', 'pygmentize', 'rst2xetex', 'rst2latex', 'uvicorn', 'rst2html4', 'activate', 'mcp-server-weibo', 'rst2s5', 'activate.nu', 'normalizer', 'rst2html5', 'deactivate.bat', 'rst2pseudoxml', 'email_validator', 'activate.csh']
# (dev) zhengnairong@xxxxxx Downloads %
```

可以看到其中的 mcp-server-weibo 是可执行文件，所以在命令行使用：

```shell
uvx --with mcp-server-weibo mcp-server-weibo
```

该例子就会出现下图：

![](../../images/2025/20251205_2_mcp_uvx.png)

## mcp-remote 系列 —— 远程托管的 MCP

```shell
"berlin-transport": {
    "command": "npx",
    "args": [
        "mcp-remote",
        "https://berlin-transport.mcp-tools.app/sse"
    ]
}
```

可以在命令行运行命令查看：

```shell
npx mcp-remote https://berlin-transport.mcp-tools.app/sse

# 结果如下：

# (dev) zhengnairong@xxxxxx Downloads % npx mcp-remote https://berlin-transport.mcp-tools.app/sse
# [26830] Using automatically selected callback port: 6245
# [26830] [26830] Connecting to remote server: https://berlin-transport.mcp-tools.app/sse
# [26830] Using transport strategy: http-first
# [26830] Received error: Error POSTing to endpoint (HTTP 404): Not Found
# [26830] Recursively reconnecting for reason: falling-back-to-alternate-transport
# [26830] [26830] Connecting to remote server: https://berlin-transport.mcp-tools.app/sse
# [26830] Using transport strategy: sse-only
# [26830] Connected to remote server using SSEClientTransport
# [26830] Local STDIO server running
# [26830] Proxy established successfully between local STDIO and remote SSEClientTransport
# [26830] Press Ctrl+C to exit
```

也可以在插件中查看：

![](../../images/2025/20251205_3_mcp_mcp-remote.png)

## smithery 系列 —— MCP 工具集市（最常见）

```shell
"Office-PowerPoint-MCP-Server": {
    "command": "npx",
    "args": [
        "-y",
        "@smithery/cli@latest",
        "run",
        "@GongRzhe/Office-PowerPoint-MCP-Server",
        "--key",
        "${SMITHERY_API_KEY}"
    ]
}
```

调用方式统一：

```shell
npx @smithery/cli run <package-name> --key <api-key> [args...]

# 例如：
# (dev) zhengnairong@xxxxxx Downloads % npx @smithery/cli run @GongRzhe/Office-PowerPoint-MCP-Server --key <my_key>
# 2025-12-05T14:43:39.447Z [Runner] Connecting to server: {"id":"@GongRzhe/Office-PowerPoint-MCP-Server","connectionTypes":["http"]}
# 2025-12-05T14:43:39.449Z [Runner] Connecting to Streamable HTTP endpoint: https://server.smithery.ai/@GongRzhe/Office-PowerPoint-MCP-Server/mcp
# 2025-12-05T14:43:39.449Z [Runner] Streamable HTTP connection initiated
# 2025-12-05T14:43:39.449Z [Runner] Streamable HTTP connection established
```

## SSE-only server（无需安装，直接 URL）

```shell
"3rd_party_mcp_server_shuidi": {
    "url": "https://mcp.shuidi.cn/sse",
    "prefer_sse": true
}
```

也可以使用插件查看：

![](../../images/2025/20251205_4_mcp_mcp-sse-only.png)

