"""
MCP Client 类实现
支持连接 npx、uvx、uv、python 等提供的 MCP server
"""

import asyncio
import httpx
import argparse
import json
from typing import Dict, Any, Optional, List
from datetime import timedelta
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.websocket import websocket_client


class MCPClient:
    """MCP 客户端类，支持多种传输方式"""

    def __init__(self, server_config: Dict[str, Any]):
        """
        初始化 MCP 客户端
        
        Args:
            server_config: MCP server 配置字典
                支持以下配置方式:
                - stdio (npx/uvx/uv/python): {"command": "npx", "args": ["mcp-server-fetch"]}
                - websocket: {"url": "ws://localhost:8080"}
                - sse: {"url": "http://localhost:8080", "prefer_sse": True}
                - streamable_http: {"url": "http://localhost:8080"}
        """
        self.server_config = server_config
        self.session: Optional[ClientSession] = None
        self._transport_context = None

    def _create_http_client(
        self,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[httpx.Timeout] = None,
        auth: Optional[httpx.Auth] = None,
    ) -> httpx.AsyncClient:
        """创建 httpx client，用于 SSE / streamable_http"""
        kwargs: Dict[str, Any] = {"follow_redirects": True}

        if timeout is None:
            kwargs["timeout"] = httpx.Timeout(30.0)
        else:
            kwargs["timeout"] = timeout

        if headers is not None:
            kwargs["headers"] = headers

        if auth is not None:
            kwargs["auth"] = auth

        return httpx.AsyncClient(**kwargs)

    def _detect_transport_type(self) -> str:
        """根据配置判断传输类型"""
        command = self.server_config.get("command")
        url = self.server_config.get("url")
        prefer_sse = self.server_config.get("prefer_sse", False)

        if command:
            return "stdio"
        if not url:
            raise ValueError("MCP server config 必须包含 'command' 或 'url'")

        if url.startswith(("ws://", "wss://")):
            return "websocket"
        if url.startswith(("http://", "https://")):
            return "sse" if prefer_sse else "streamable_http"

        raise ValueError(f"不支持的 URL 协议: {url}")

    @asynccontextmanager
    async def _open_transport(self):
        """建立底层传输连接"""
        transport_type = self._detect_transport_type()

        if transport_type == "stdio":
            if not self.server_config.get("command"):
                raise ValueError("STDIO 传输需要 'command'")

            # 获取环境变量，如果用户没有提供则使用空字典
            env = self.server_config.get("env")
            if env is None:
                env = {}
            else:
                # 复制一份，避免修改原始配置
                env = dict(env)

            # 对于 uvx/uv 命令，自动添加 UV_QUIET 环境变量来抑制输出
            # 这样可以避免非 JSON 输出（如 "found 0 vulnerabilities"）被误解析为 JSONRPC 消息
            command = self.server_config["command"]
            if command in ("uvx", "uv") and "UV_QUIET" not in env:
                env["UV_QUIET"] = "1"

            params = StdioServerParameters(
                command=command,
                args=self.server_config.get("args", []) or [],
                env=env if env else None,
                cwd=self.server_config.get("cwd"),
                encoding=self.server_config.get("encoding", "utf-8"),
            )

            async with stdio_client(params) as (read_stream, write_stream):
                yield read_stream, write_stream

        elif transport_type == "sse":
            url = self.server_config.get("url")
            if not url:
                raise ValueError("SSE 传输需要 'url'")

            timeout = float(self.server_config.get("timeout", 30.0))
            sse_read_timeout = float(self.server_config.get("sse_read_timeout", 300.0))

            try:
                async with sse_client(
                    url=url,
                    headers=self.server_config.get("headers"),
                    timeout=timeout,
                    sse_read_timeout=sse_read_timeout,
                    httpx_client_factory=self._create_http_client,
                ) as (read_stream, write_stream):
                    yield read_stream, write_stream
            except TypeError:
                # 兼容老版本 mcp
                async with sse_client(
                    url=url,
                    headers=self.server_config.get("headers"),
                    timeout=timeout,
                    sse_read_timeout=sse_read_timeout,
                ) as (read_stream, write_stream):
                    yield read_stream, write_stream

        elif transport_type == "streamable_http":
            url = self.server_config.get("url")
            if not url:
                raise ValueError("StreamableHTTP 传输需要 'url'")

            timeout = float(self.server_config.get("timeout", 30.0))
            sse_read_timeout = float(self.server_config.get("sse_read_timeout", 300.0))
            terminate_on_close = bool(self.server_config.get("terminate_on_close", True))

            try:
                async with streamablehttp_client(
                    url=url,
                    headers=self.server_config.get("headers"),
                    timeout=timedelta(seconds=timeout),
                    sse_read_timeout=timedelta(seconds=sse_read_timeout),
                    terminate_on_close=terminate_on_close,
                    httpx_client_factory=self._create_http_client,
                ) as (read_stream, write_stream, _get_session_id):
                    yield read_stream, write_stream
            except TypeError:
                async with streamablehttp_client(
                    url=url,
                    headers=self.server_config.get("headers"),
                    timeout=timedelta(seconds=timeout),
                    sse_read_timeout=timedelta(seconds=sse_read_timeout),
                    terminate_on_close=terminate_on_close,
                ) as (read_stream, write_stream, _get_session_id):
                    yield read_stream, write_stream

        elif transport_type == "websocket":
            url = self.server_config.get("url")
            if not url:
                raise ValueError("WebSocket 传输需要 'url'")

            async with websocket_client(url=url) as (read_stream, write_stream):
                yield read_stream, write_stream

        else:
            raise ValueError(f"未知传输类型: {transport_type}")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def connect(self):
        """连接到 MCP server"""
        if self.session is not None:
            return

        read_timeout = float(self.server_config.get("timeout", 30.0))
        self._transport_context = self._open_transport()
        read_stream, write_stream = await self._transport_context.__aenter__()

        self.session = ClientSession(
            read_stream=read_stream,
            write_stream=write_stream,
            read_timeout_seconds=timedelta(seconds=read_timeout),
        )
        await self.session.__aenter__()
        await self.session.initialize()

    async def close(self):
        """关闭连接"""
        if self.session is not None:
            await self.session.__aexit__(None, None, None)
            self.session = None

        if self._transport_context is not None:
            await self._transport_context.__aexit__(None, None, None)
            self._transport_context = None

    async def list_tools(self) -> List[Any]:
        """获取可用工具列表"""
        if self.session is None:
            raise RuntimeError("未连接，请先调用 connect() 或使用 async with")

        tools_result = await self.session.list_tools()
        return tools_result.tools if tools_result else []

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """调用工具"""
        if self.session is None:
            raise RuntimeError("未连接，请先调用 connect() 或使用 async with")

        return await self.session.call_tool(tool_name, arguments=arguments or {})


def get_mcp_servers(server_type: str = "uvx"):
    """
    配置 MCP server（支持多种传输方式）
    
    Args:
        server_type: 服务器类型
            - "uvx": 使用 uvx 启动 (stdio)
            - "npx": 使用 npx 启动 (stdio)
            - "uv": 使用 uv 启动 (stdio)
            - "python": 使用 python 启动 (stdio)
            - "websocket": WebSocket 传输
            - "sse": SSE 传输
    
    Returns:
        MCP server 配置字典
    """
    configs = {
        # stdio 传输方式
        "uvx": {
            "command": "uvx",
            "args": ["mcp-server-fetch"],
        },
        "npx": {
            "command": "npx",
            "args": ["-y", "wikipedia-mcp-server"],
        },
        "uv": {
            "command": "uv",
            "args": ["run", "--with", "biomcp-python", "biomcp", "run"],
        },
        "python": {
            "command": "python",
            "args": ["-m", "mcp_server_fetch"],
        },
        # WebSocket 传输
        "websocket": {
            "url": "ws://localhost:8080/mcp",
        },
        # SSE 传输
        "sse": {
            "url": "https://mcp.shuidi.cn/sse",
            "prefer_sse": True,
        },
        # Streamable HTTP 传输
        "streamable_http": {
            "url": "http://localhost:8080/mcp",
            "headers": {
                "Authorization": "Bearer your-token-here",
            },
        },
    }
    
    if server_type not in configs:
        raise ValueError(f"不支持的服务器类型: {server_type}。可选: {list(configs.keys())}")
    
    return configs[server_type]


def get_tool_call_arguments(server_type: str = "uvx"):
    """
    获取工具调用参数
    根据不同的 server_type 返回相应的工具调用参数
    
    注意: 工具名称和参数需要根据实际 MCP server 提供的工具进行调整。
    可以使用 --list_tools 参数查看实际可用的工具名称和参数。
    
    Args:
        server_type: 服务器类型
    
    Returns:
        dict: 工具调用参数字典
            "工具名称": {
                "参数名": "参数值",
                ...
            }
    """
    if server_type == "uvx":
        # uvx 使用 mcp-server-fetch，提供 fetch 工具
        tool_call_arguments = {
            "fetch": {
                "url": "https://modelcontextprotocol.io/docs/develop/build-client",
            }
        }
    elif server_type == "npx":
        # npx 使用 wikipedia-mcp-server，提供搜索维基百科的工具
        tool_call_arguments = {
            "wikipedia_search": {
                "query": "Python programming language",
            }
        }
    elif server_type == "uv":
        # uv 使用 biomcp，提供生物信息学相关的工具
        tool_call_arguments = {
            "search": {
                "query": "human",
                "limit": 10,
            }
        }
    elif server_type == "python":
        # python 使用 mcp_server_fetch，和 uvx 类似
        tool_call_arguments = {
            "fetch": {
                "url": "https://modelcontextprotocol.io/docs/develop/build-client",
            }
        }
    elif server_type == "sse":
        # SSE 传输，使用企业数据查询工具
        tool_call_arguments = {
            "query_company_data": {
                "question": "按省统计2024年成立的企业，并以地图可视化展示",
            }
        }
    elif server_type == "websocket":
        # WebSocket 传输，需要根据实际服务器配置
        # 这里提供一个通用示例，实际使用时需要根据服务器提供的工具调整
        tool_call_arguments = {
            "example_tool": {
                "param1": "value1",
            }
        }
    elif server_type == "streamable_http":
        # Streamable HTTP 传输，需要根据实际服务器配置
        # 这里提供一个通用示例，实际使用时需要根据服务器提供的工具调整
        tool_call_arguments = {
            "example_tool": {
                "param1": "value1",
            }
        }
    else:
        # 默认情况
        tool_call_arguments = {}

    return tool_call_arguments


async def main(args: argparse.Namespace):
    """示例：使用 MCPClient 连接并调用 MCP server"""
    server_config = get_mcp_servers(args.server_type)
    print(f"使用服务器类型: {args.server_type}")
    print(f"配置: {json.dumps(server_config, indent=2, ensure_ascii=False)}")
    
    async with MCPClient(server_config) as client:
        if args.list_tools:
            # 列出所有可用工具
            tools = await client.list_tools()
            print(f"\n可用工具数量: {len(tools)}")
            for tool in tools:
                print(f"- {tool.name}")
                tool_dump = tool.model_dump()
                # print(f"  全信息: {json.dumps(tool_dump, ensure_ascii=False)}")
            return

        # 调用工具
        tool_call_arguments = get_tool_call_arguments(args.server_type)
        
        if not tool_call_arguments:
            print("警告: 未配置工具调用参数，请先使用 --list_tools 查看可用工具，然后修改 get_tool_call_arguments 函数")
            return
        
        # 先获取可用工具列表，用于验证工具是否存在
        available_tools = await client.list_tools()
        available_tool_names = {tool.name for tool in available_tools}
        
        for tool_name, tool_arguments in tool_call_arguments.items():
            if tool_name not in available_tool_names:
                print(f"警告: 工具 '{tool_name}' 不存在。可用工具: {', '.join(available_tool_names)}")
                print("提示: 请使用 --list_tools 查看所有可用工具，然后修改 get_tool_call_arguments 函数中的工具名称")
                continue
            
            try:
                print(f"\n调用工具: {tool_name}")
                print(f"参数: {tool_arguments}")
                result = await client.call_tool(tool_name, arguments=tool_arguments)
                print(f"调用结果: {result}")
            except Exception as e:
                print(f"调用失败: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Client 示例")
    parser.add_argument(
        "--server_type",
        type=str,
        default="sse",
        choices=["uvx", "npx", "uv", "python", "websocket", "sse", "streamable_http"],
        help="MCP server 类型",
    )
    parser.add_argument(
        "--list_tools",
        action="store_true",
        default=False,
        help="列出所有可用工具",
    )
    args = parser.parse_args()
    asyncio.run(main(args))

