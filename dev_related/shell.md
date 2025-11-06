- [基础shell语法](#基础shell语法)
- [regexp](#regexp)
- [cURL](#curl)
- [wscat](#wscat)
- [nc (netcat)](#nc-netcat)

# 基础shell语法

**bash特殊变量**

| 特殊变量      | 含义                                 | 示例                              | 输出说明                                        |
| ------------- | ------------------------------------ | --------------------------------- | ----------------------------------------------- |
| `$0`          | 当前脚本名称                         | `echo $0`                         | 如果运行 `./myscript.sh` → 输出 `./myscript.sh` |
| `$1`、`$2`... | 传入脚本的第 1、2、3 个参数          | `echo $1 $2`                      | 脚本参数                                        |
| `$#`          | 参数个数（不含脚本名）               | `echo $#`                         | 传入参数的数量                                  |
| `$@`          | 所有参数（独立展开，每个为单独参数） | `for a in "$@"; do echo $a; done` | 每个参数是独立的字符串（`"A" "B" "C"`）         |
| `$*`          | 所有参数（整体展开）                 | `echo "$*"`                       | 所有参数合并为一个字符串（`"A B C"`）           |
| `$?`          | 上一个命令的退出状态码（0=成功）     | `ls /not_exist; echo $?`          | 输出 2（表示错误）                              |
| `$$`          | 当前脚本的进程号（PID）              | `echo $$ `                        | 进程 ID（如 12345）                             |
| `$!`          | 最近一个后台任务的 PID               | `sleep 10 & echo $!`              | 输出后台任务的 PID                              |
| `$_`          | 上一条命令的最后一个参数             | `echo hello world; echo $_`       | 输出 world                                      |
| `$-`          | 当前 shell 的选项标志                | `echo $-`                         | 如 himBH，代表启用的模式                        |
| `$PWD`        | 当前工作目录路径                     | `echo $PWD`                       | 输出 `/home/user`                               |
| `$OLDPWD`     | 上一次所在目录                       | `cd /tmp; echo $OLDPWD`           | 输出切换前的路径                                |
| `$USER`       | 当前用户名                           | `echo $USER`                      | 输出用户名                                      |
| `$HOME`       | 当前用户主目录                       | `echo $HOME`                      | 输出用户主目录                                  |
| `$PATH`       | 系统可执行程序路径列表               | `echo $PATH`                      | 多个路径用冒号分隔                              |
| `$HOSTNAME`   | 主机名                               | `echo $HOSTNAME`                  | 输出机器名称                                    |
| `$RANDOM`     | 生成一个 0–32767 的随机整数          | `echo $RANDOM`                    | 每次运行不同                                    |
| `$LINENO`     | 当前脚本的行号                       | `echo $LINENO`                    | 输出脚本执行到的行号                            |
| `$SECONDS`    | 脚本运行的秒数（从启动算起）         | `sleep 2; echo $SECONDS`          | 输出 2                                          |
| `$FUNCNAME`   | 当前函数名                           | 在函数内用 `echo $FUNCNAME`       | 输出函数名                                      |

```shell
#!/bin/bash
# ======================================================
# 文件名: test.sh
# 作用: 演示 Bash 特殊变量的使用与效果
# 运行命令：bash test.sh hello world
# ======================================================

echo "===== 🧩 基本信息 ====="
echo "脚本名称 (\$0): $0"
echo "脚本进程号 (\$$): $$"
echo "当前用户 (\$USER): $USER"
echo "主机名 (\$HOSTNAME): $HOSTNAME"
echo "当前目录 (\$PWD): $PWD"
echo "用户主目录 (\$HOME): $HOME"
echo

echo "===== 📦 参数信息 ====="
echo "传入的参数数量 (\$#): $#"
echo "所有参数整体 (\$*): $*"
echo "所有参数分开 (\$@): $@"
echo "第一个参数 (\$1): $1"
echo "第二个参数 (\$2): $2"
echo

echo "===== 🔁 遍历参数 (\$@ vs \$*) ====="
echo "使用 \"\$@\": 每个参数独立处理"
for arg in "$@"; do
    echo " -> 参数: [$arg]"
done

echo
echo "使用 \"\$*\": 参数被视为一个整体"
for arg in "$*"; do
    echo " -> 参数: [$arg]"
done
echo

echo "===== ⚙️ 上一个命令状态 ====="
ls /not_exist_dir > /dev/null 2>&1
echo "上一个命令返回值 (\$?): $?"
echo

echo "===== 🧠 后台任务相关 ====="
(sleep 3) &
echo "后台任务 PID (\$!): $!"
echo

echo "===== 🎲 随机数、时间、行号 ====="
echo "随机数 (\$RANDOM): $RANDOM"
sleep 1
echo "脚本已运行时间 (\$SECONDS): $SECONDS 秒"
echo "当前脚本行号 (\$LINENO): $LINENO"
echo

echo "===== 🔍 上一个命令最后一个参数 (\$_) ====="
echo "上一条命令最后一个参数是: [$_]"
echo

echo "===== ✅ 当前 shell 选项 (\$-) ====="
echo "当前启用的 shell 选项标志: $-"
echo

echo "===== 🧾 函数中的特殊变量 ====="
show_func_info() {
    echo "函数名 (\$FUNCNAME): $FUNCNAME"
    echo "函数第一个参数 (\$1): $1"
    echo "函数被调用时的总参数数 (\$#): $#"
}
show_func_info "demo_arg"

echo
echo "===== ✅ 演示结束 ====="
```

**变量与命令替换**

| 语法         | 说明                 | 示例              |
| ------------ | -------------------- | ----------------- |
| `VAR=value`  | 定义变量             | `name="Alice"`    |
| `$VAR`       | 使用变量             | `echo $name`      |
| `${VAR}`     | 更安全的变量引用     | `echo ${name}_ok` |
| `$(command)` | **执行命令并取结果** | `now=$(date)`     |
| `command`    | 老式写法（不推荐）   | now=`date`        |

**数值与字符串比较**

| 比较符    | 说明       | 示例               |
| --------- | ---------- | ------------------ |
| `-eq`     | 等于       | `[ $a -eq $b ]`    |
| `-ne`     | 不等于     | `[ $a -ne $b ]`    |
| `-lt`     | 小于       | `[ $a -lt $b ]`    |
| `-le`     | 小于等于   | `[ $a -le $b ]`    |
| `-gt`     | 大于       | `[ $a -gt $b ]`    |
| `-ge`     | 大于等于   | `[ $a -ge $b ]`    |
| `=`       | 字符串相等 | `[ "$a" = "$b" ]`  |
| `!=`      | 字符串不等 | `[ "$a" != "$b" ]` |
| `-z "$a"` | 字符串为空 | `[ -z "$a" ]`      |
| `-n "$a"` | 字符串非空 | `[ -n "$a" ]`      |

**文件或目录判断**

| 选项 | 含义                       | 示例              |
| ---- | -------------------------- | ----------------- |
| `-d` | 是否为目录                 | `[ -d "$DIR" ]`   |
| `-f` | 是否为普通文件             | `[ -f "$FILE" ]`  |
| `-e` | 是否存在（文件或目录都算） | `[ -e "$PATH" ]`  |
| `-s` | 文件是否存在且非空         | `[ -s "$FILE" ]`  |
| `-r` | 文件是否有可读权限         | `[ -r "$FILE" ]`  |
| `-w` | 文件是否有可写权限         | `[ -w "$FILE" ]`  |
| `-x` | 文件是否有可执行权限       | `[ -x "$FILE" ]`  |
| `!`  | 取反（not）                | `[ ! -d "$DIR" ]` |


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


# nc (netcat)

**常用参数**

| 参数 | 说明                             |
| ---- | -------------------------------- |
| `-l` | 启动监听（Listen mode）          |
| `-p` | 指定本地端口号                   |
| `-v` | 显示详细信息（verbose）          |
| `-n` | 不进行 DNS 解析（加快速度）      |
| `-z` | 只扫描端口，不发送数据           |
| `-u` | 使用 UDP（默认 TCP）             |
| `-w` | 设置超时秒数                     |
| `-k` | 保持监听，多次接收连接（若支持） |

**端口扫描（快速探测某台主机哪些端口开放）**

```shell
# 1. 简单测试
nc -zv ${ip} ${port}
nc -zv localhost 1-1024 # 探测本机 1-1024 端口（localhost一般会扫描ipv4跟ipv6两种，所以会有两条log）

# 2. 更细粒度脚本（显示开放端口）
for p in {1..1024}; do
  nc -z -w 1 localhost $p 2>/dev/null && echo "open $p"
done
```

**简单通信 / 聊天（两个终端互相发消息）**

```shell
# 服务端
nc -lk ${listen_port}
# 客户端
nc ${server_ip} ${port}

# 然后就可以互相发送消息
```

**模拟http交互**

模拟与curl的交互

```shell
# 服务端
nc -lk ${listen_port}
# 客户端
curl http://${server_ip}:${port}

# 然后服务端会有类似以下的log：
    # GET / HTTP/1.1
    # Host: localhost:60012
    # User-Agent: curl/8.7.1
    # Accept: */*
# 可以在服务端模拟返回：
    # HTTP/1.1 200 OK
    # Content-Type: text/plain
    # Content-Length: 13

    # Hello, world!
# 客户端就可以收到返回
# 当然，你还可以测试json之类的返回格式
```

http当服务端，nc当客户端

```shell
# 服务端
python -m http.server ${port}
# 客户端
nc ${server_ip} ${port}

# 然后客户端用一下信息模拟请求（要有空行，也就是多按个回车即可）：
    # GET / HTTP/1.1
    # Host: localhost:8000
    # Connection: close
    # 
# 就会收到类似以下的log：
    # HTTP/1.0 200 OK
    # Server: SimpleHTTP/0.6 Python/3.12.11
    # Date: Thu, 06 Nov 2025 04:58:16 GMT
    # Content-type: text/html; charset=utf-8
    # Content-Length: 308

    # <!DOCTYPE HTML>
    # <html lang="en">
    # <head>
    # <meta charset="utf-8">
    # <title>Directory listing for /</title>
    # </head>
    # <body>
    # <h1>Directory listing for /</h1>
    # <hr>
    # <ul>
    # <li><a href="test.json">test.json</a></li>
    # <li><a href="test.py">test.py</a></li>
    # <li><a href="test.sh">test.sh</a></li>
    # </ul>
    # <hr>
    # </body>
    # </html>
```
