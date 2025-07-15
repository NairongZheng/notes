- [regexp](#regexp)
- [cURL](#curl)
- [wscat](#wscat)


# regexp

| 符号    | 含义                                          | 示例                                       |
| ------- | --------------------------------------------- | ------------------------------------------ |
| `.`     | 匹配任意单个字符（除了换行符）                | `a.b` 匹配 `acb`, `arb` 等                 |
| `^`     | 匹配行首                                      | `^Hello` 匹配以 `Hello` 开头的行           |
| `$`     | 匹配行尾                                      | `world$` 匹配以 `world` 结尾的行           |
| `*`     | 匹配前一个字符重复 0 次或多次                 | `lo*` 匹配 `l`, `lo`, `loo`, ...           |
| `+`     | 匹配前一个字符重复 1 次或多次（至少匹配一次） | `lo+` 匹配 `lo`, `loo`, `looo`，不匹配 `l` |
| `?`     | 匹配前一个字符 0 次或 1 次                    | `colou?r` 匹配 `color` 或 `colour`         |
| `[]`    | 字符集，匹配其中任一字符                      | `[aeiou]` 匹配任一元音字母                 |
| `[^]`   | 否定字符集                                    | `[^0-9]` 匹配非数字字符                    |
| `\`     | 转义特殊字符                                  | `\.` 匹配字面意义的 `.`                    |
| `()`    | 分组                                          | `(ab)+` 匹配 `ab`, `abab`, `ababab`        |
| `{n}`   | 恰好重复 n 次                                 | `a{3}` 匹配 `aaa`                          |
| `{n,}`  | 至少重复 n 次                                 | `a{2,}` 匹配 `aa`, `aaa`, `aaaa`           |
| `{n,m}` | 重复 n 到 m 次                                | `a{2,4}` 匹配 `aa`, `aaa`, `aaaa`          |
| `\d`    | 匹配任意数字字符，等价于 `[0-9]`              |                                            |
| `\w`    | 匹配字母、数字或下划线 `[a-zA-Z0-9_]`         |                                            |
| `\s`    | 匹配空白字符（空格、制表符、换行等）          |                                            |


# cURL

**-v 显示详细的信息**

```bash
# -v（verbose）：显示详细的信息，包括请求头、响应头和传输过程等。
curl -v https://httpbin.org
```

**-I 与 -i**

```bash
# 只显示响应体
curl https://httpbin.org

# -I（head）：只显示响应头，不显示响应体。不发起完整 GET 请求，只发 HEAD 请求，因此响应体本身不会发送
curl -I https://httpbin.org

# -i（include）：包含响应头信息，也会显示响应体。
curl -i https://httpbin.org
```

**-X 指定请求方法**

```bash
# -X：指定请求方法，可以指定请求的方法，如 GET、POST、PUT、DELETE 等。
curl -X POST https://httpbin.org/post
```

**-d 发送数据**

```bash
# -d：发送数据，会自动变成 POST 请求
curl -X POST -d "name=Tom&age=25" https://httpbin.org/post  # 发送表单数据
```

**-H 设置自定义请求头**

```bash
curl -X POST -H "Content-Type: application/json" -d '{"name": "damon"}' https://httpbin.org/post
    # 设置了 Content-Type 为 application/json，并发送一个 JSON 格式的数据作为请求体。
```

**-A 设置 User-Agent**

```bash
# -A 或 --user-agent：设置请求头中的 User-Agent，这个字段用来表示使用的客户端（浏览器、爬虫、curl等）
# 服务器有时会根据这个字段 决定是否响应请求、返回哪种页面、是否阻止访问。
curl -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/90.0" https://httpbin.org/user-agent
    # Chrome 浏览器：Mozilla/5.0 (Windows NT 10.0; Win64; x64)...
    # curl：curl/7.64.1
    # Python requests：python-requests/2.25.1
```

**-o 将响应输出保存到文件中**

```bash
# -o（output）：将响应内容保存到指定的文件中，文件名由用户自定义。
curl -o image.png https://httpbin.org/image/png
```

**-O 以 URL 中的文件名保存**

```bash
# -O（remote-name）：将响应内容保存到远程服务器提供的文件名中。
curl -O https://httpbin.org/image/png
```

**-u 使用用户名和密码**

```bash
# -u：使用用户名和密码，用于访问需要基本认证的 API。
curl -u user:passwd https://httpbin.org/basic-auth/user/passwd
```

**-L 跟踪重定向**

```bash
# -L：跟踪重定向，自动跳转到最终页面。
curl -L http://httpbin.org/redirect/1
```

**-s 静默模式**

```bash
# -s：静默模式，不显示进度条或错误，适合脚本中使用。
curl -s -o img.png https://httpbin.org/image/png
```

**组合使用**

```bash
curl -X POST https://httpbin.org/post \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"1234"}'
```

# wscat

**安装**

```bash
# 确保你已安装 Node.js（包含 npm）
node -v
npm -v
# 安装wscat（若无）
npm install -g wscat
wscat -V
```

**基本用法：连接 WebSocket 服务器**

```bash
# 连接到 WebSocket 服务器
wscat -c ws://echo.websocket.org
```

**-c 指定连接地址**

```bash
# -c（--connect）：指定要连接的 WebSocket 地址
wscat -c ws://localhost:8080
```

**-H 设置自定义请求头**

```bash
# -H：设置自定义请求头（可多次使用）
wscat -c ws://localhost:8080 -H "Authorization: Bearer <token>"
```

**-p 指定子协议**

```bash
# -p：指定 WebSocket 子协议
wscat -c ws://localhost:8080 -p "chat"
```

**-o 以只读模式连接**

```bash
# -o：只读模式，只接收消息不发送
wscat -c ws://localhost:8080 -o
```

**-n 禁用颜色输出**

```bash
# -n：禁用彩色输出
wscat -c ws://localhost:8080 -n
```

**发送消息**

连接后，直接输入内容并回车即可发送消息到服务器。

**组合使用**

```bash
wscat -c ws://localhost:8080 -H "Authorization: Bearer <token>" -p "chat"
```
