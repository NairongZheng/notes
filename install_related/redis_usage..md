

## 安装并配置redis

```bash
# 安装redis
apt install -y redis-server
# 启动redis（配置文件路径: /etc/redis/redis.conf）
systemctl start redis   # 或者: service redis-server start
# 测试redis
redis-cli ping          # 应该返回 PONG
redis-cli               # 可以看到提示符变为 127.0.0.1:6379>
```

## 设置需要tls认证的redis

若不需要认证，可以忽略这一步！！！

**升级redis（tls要求redis版本大于6）**

```bash

# 若系统版本不够，可以用以下方式升级
apt update
apt install -y software-properties-common
add-apt-repository ppa:redislabs/redis
apt update
apt install -y redis-server
# 系统可能有多个redis，比如我就有：
    # /opt/anaconda3/envs/dev/bin/redis-server
    # /usr/bin/redis-server
# 需要设置好使用哪个redis-server，查看版本
redis-server --version
```

**证书生成（使用openssl）**

```bash
mkdir ~/redis-tls && cd ~/redis-tls
# 1. 生成 CA 私钥和证书
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -out ca.crt -subj "/CN=Redis Test CA"
# 2. 生成服务器私钥和证书请求（CSR）
openssl genrsa -out redis.key 2048
openssl req -new -key redis.key -out redis.csr -subj "/CN=redis-server"
# 3. 用 CA 签发服务器证书
openssl x509 -req -in redis.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out redis.crt -days 3650 -sha256

# 最终你会有以下文件：
    # ca.crt: 根证书（客户端需要）
    # redis.crt: 服务器证书
    # redis.key: 服务器私钥
```

**配置 redis.conf**

```bash
vim /etc/redis/redis.conf

# 添加以下内容

# 启用 TLS 并监听 6379 TLS 端口
tls-port 6379
# 关闭明文端口
port 0
# TLS 相关路径（使用绝对路径）
tls-cert-file /root/redis-tls/redis.crt
tls-key-file /root/redis-tls/redis.key
tls-ca-cert-file /root/redis-tls/ca.crt
# 可以加上密码认证
requirepass <your_redis_password_here>

# 重启redis-server
systemctl restart redis  # 或者: service redis-server restart
```

## 命令行使用redis

**进入redis**

```bash
# 若无密码无认证
redis-cli
# 若有密码有认证：先认证进入，再输密码（直接命令行 -a 输入密码不安全）
redis-cli --tls --cert /root/redis-tls/redis.crt --key /root/redis-tls/redis.key --cacert /root/redis-tls/ca.crt
# 会有提示符 127.0.0.1:6379>
# 然后使用 auth <your_redis_password_here> 进行密码认证
# 命令行结果如下：
# [root@b2430d6331e1:/data/p3/damonzheng/tmp/code]
# $ redis-cli --tls --cert /root/redis-tls/redis.crt --key /root/redis-tls/redis.key --cacert /root/redis-tls/ca.crt
# 127.0.0.1:6379> auth damonzheng
# OK
# 127.0.0.1:6379>
```

**常用命令**

```bash
# 1. 连接与信息相关
redis-cli                       # 进入 Redis 命令行客户端。
PING                            # 测试连接，返回 PONG 表示服务器在线。
INFO                            # 查看 Redis 服务器的详细信息（版本、内存、连接数等）。
CLIENT LIST                     # 查看当前所有客户端连接信息。

# 2. 键操作
KEYS *                          # 查询所有 key（生产环境慎用，可能阻塞）。
EXISTS <key>                    # 判断 key 是否存在，存在返回 1，不存在返回 0。
DEL <key>                       # 删除指定 key。
EXPIRE <key> seconds            # 设置 key 的过期时间（秒）。
TTL <key>                       # 查看 key 剩余过期时间（秒）。

# 3. 字符串操作
SET <key> <value> [EX seconds]  # 设置 key 对应的字符串值，可以带过期时间。
GET <key>                       # 获取 key 的字符串值。
INCR <key>                      # key 的整数值加 1。
DECR <key>                      # key 的整数值减 1。
MGET <key1> <key2> ...          # 一次获取多个 key 的值。

# 4. 哈希（Hash）操作
HSET <key> <field> <value       # 设置哈希表 key 中字段 field 的值。
HGET <key> <field>              # 获取哈希表 key 中字段 field 的值。
HDEL <key> <field>              # 删除哈希表 key 中的字段 field。
HGETALL <key>                   # 获取哈希表 key 中所有字段和值。

# 5. 列表（List）操作
LPUSH key value                 # 从左边压入一个值到列表。
RPUSH key value                 # 从右边压入一个值到列表。
LPOP key                        # 从左边弹出一个值。
RPOP key                        # 从右边弹出一个值。
LRANGE key start stop           # 获取列表指定范围内的元素（start、stop 支持负数）。

# 6. 集合（Set）操作
SADD key member                 # 向集合添加一个成员。
SREM key member                 # 从集合移除一个成员。
SMEMBERS key                    # 获取集合所有成员。
SISMEMBER key member            # 判断成员是否在集合中。

# 7. 其他
SELECT <db_num>                 # 选择数据库（默认 0-15 共16个）。
FLUSHDB                         # 清空当前数据库所有数据。
FLUSHALL                        # 清空所有数据库数据。
```

## python使用redis示例

```python
import redis
import json
import time
from typing import Any
import ssl


ssl_context = ssl.create_default_context(
    cafile="/root/redis-tls/ca.crt"
)
ssl_context.load_cert_chain(
    certfile="/root/redis-tls/redis.crt",
    keyfile="/root/redis-tls/redis.key"
)

# 自定义异常
class CacheMiss(Exception):
    pass

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            ssl=True,
            # ssl_context=ssl_context, # 新版本可用，旧版本使用下面的方式
            ssl_certfile='/root/redis-tls/redis.crt',
            ssl_keyfile='/root/redis-tls/redis.key',
            ssl_ca_certs='/root/redis-tls/ca.crt',
            password="damonzheng"
        )

    def set(self, key: str, value: Any, ttl: int = 60):
        """
        存储一个键值对，值将被序列化为 JSON 字符串
        :param key: 键
        :param value: 任意可序列化对象
        :param ttl: 过期时间（秒）
        """
        try:
            json_value = json.dumps(value)
            self.client.set(name=key, value=json_value, ex=ttl)
        except Exception as e:
            raise RuntimeError(f"Set error: {e}")

    def get(self, key: str) -> Any:
        """
        获取一个键对应的值，如果不存在则抛出 CacheMiss
        :param key: 键
        :return: 反序列化后的对象
        """
        raw = self.client.get(key)
        if raw is None:
            raise CacheMiss(f"Key not found: {key}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    def delete(self, key: str):
        self.client.delete(key)

    def ttl(self, key: str) -> int:
        return self.client.ttl(key)

# 示例使用
if __name__ == "__main__":
    cache = RedisCache()

    user_data = {"id": 1, "name": "Alice", "age": 25}
    key = "user:1"

    print("▶ 设置缓存")
    cache.set(key, user_data, ttl=10)

    print("▶ 获取缓存")
    try:
        value = cache.get(key)
        print("缓存值：", value)
    except CacheMiss:
        print("缓存未命中")

    print("▶ TTL 剩余：", cache.ttl(key), "秒")

    print("▶ 等待缓存过期...")
    time.sleep(11)

    try:
        cache.get(key)
    except CacheMiss:
        print("缓存过期后获取失败：CacheMiss")
```