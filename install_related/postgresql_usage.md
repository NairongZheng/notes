- [安装](#安装)
- [命令行使用postgre](#命令行使用postgre)
- [python使用postgre示例](#python使用postgre示例)
- [使用 pgAdmin 可视化管理 PostgreSQL](#使用-pgadmin-可视化管理-postgresql)


## 安装

```bash
# 安装postgre
apt update
apt install -y postgresql postgresql-contrib
# 启动postgre
systemctl start postgresql   # 或者: service postgresql start
```

## 命令行使用postgre

**管理员进入数据库**

```bash
# 1.切换到postgres用户并进入数据库命令行（postgres用户是postgres的超级管理员！）
sudo -i -u postgres
psql

# 2. 直接使用postgres进入（跟上面一个意思）
sudo -u postgres psql
```

**创建数据库与用户**

```sql
-- 创建数据库 testdb
CREATE DATABASE testdb;
-- 创建用户 testuser 并设置密码为 password
CREATE USER testuser WITH ENCRYPTED PASSWORD 'password';
-- 授予数据库权限（数据库级权限）
GRANT ALL PRIVILEGES ON DATABASE testdb TO testuser;

-- 切换数据库
\c testdb

-- 创建表 表名users
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    age INT
);

-- 插入数据
INSERT INTO users (name, age) VALUES ('Alice', 25);

-- 将表的 SELECT 权限授予 testuser（表级权限）
GRANT SELECT ON TABLE users TO testuser;
```

**常用命令**

```bash
\conninfo                           # 显示你当前连接的数据库名、用户名、主机等信息。
\l                                  # 列出所有数据库
\c <dbname>                         # 连接到数据库
\du                                 # 查看所有用户
\dt                                 # 查看所有表
\d <tablename>                      # 查看表结构
\q                                  # 退出

# 更详细的查询就需要用spl语言了

# postgres=# \dt
#          List of relations
#  Schema | Name  | Type  |  Owner
# --------+-------+-------+----------
#  public | users | table | postgres
# (1 row)

# postgres=# \d users
#                             Table "public.users"
#  Column |  Type   | Collation | Nullable |              Default
# --------+---------+-----------+----------+-----------------------------------
#  id     | integer |           | not null | nextval('users_id_seq'::regclass)
#  name   | text    |           | not null |
#  age    | integer |           |          |
# Indexes:
#     "users_pkey" PRIMARY KEY, btree (id)
```

**使用非超级用户登录数据库**

默认情况下，PostgreSQL 使用 “peer” 认证，这会导致非 postgres 用户登录失败。（所以一般都用超级管理员postgres就好）

非超级用户可以直接使用以下命令来登录：

```bash
psql -U <username> -d <db_name> -h 127.0.0.1        # 使用username输入密码进入db_name数据库
```

至于用户的登录验证方式是在配置文件中设置的。

配置文件位置可以在 psql 中使用以下命令查看：

```sql
SHOW config_file;   -- postgresql.conf 文件路径
SHOW hba_file;      -- pg_hba.conf 文件路径
SHOW ident_file;    -- pg_ident.conf 文件路径
SHOW data_directory;-- 数据目录路径
```

其中的 `pg_hba.conf` 文件中就存有用户登录的验证方式配置。示例如下：

```bash
# Database administrative login by Unix domain socket
local   all             postgres                                peer

# TYPE  DATABASE        USER            ADDRESS                 METHOD

# "local" is for Unix domain socket connections only
local   all             all                                     peer
# IPv4 local connections:
host    all             all             127.0.0.1/32            md5
# IPv6 local connections:
host    all             all             ::1/128                 md5
# Allow replication connections from localhost, by a user with the
# replication privilege.
local   replication     all                                     peer
host    replication     all             127.0.0.1/32            md5
host    replication     all             ::1/128                 md5
```


## python使用postgre示例

```python
# 前提是在postgre中创建了testdb数据库跟testuser用户，并且在其中建立了user表格
# 并且授权testdb给testuser，并且给了SELECT权限

import psycopg2

conn = psycopg2.connect(
    dbname="testdb",
    user="testuser",
    password="password",
    host="localhost",
    port="5432"
)

cur = conn.cursor()
cur.execute("SELECT * FROM users;")
rows = cur.fetchall()
for row in rows:
    print(row)

cur.close()
conn.close()
```


## 使用 pgAdmin 可视化管理 PostgreSQL

**启动pgAdmin**

可以用apt，但是比较推荐用docker：

```bash
docker run -p 5050:80 \
    -e PGADMIN_DEFAULT_EMAIL=<email> \
    -e PGADMIN_DEFAULT_PASSWORD=<password> \
    --name pgadmin \
    -d dpage/pgadmin4

# 然后访问 http://localhost:5050，输入邮箱跟密码即可登录
```

**连接 PostgreSQL 数据库**

1. 左侧点击 `Add New Server`
2. 配置连接参数，包括general选项卡、连接选项卡等。