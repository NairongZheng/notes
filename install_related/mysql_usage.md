
- [安装](#安装)
- [命令行使用mysql](#命令行使用mysql)
- [常用sql语句](#常用sql语句)
- [python使用mysql](#python使用mysql)


## 安装

```bash
# 安装 MySQL
apt update
apt install -y mysql-server
# 启动 MySQL 服务
systemctl start mysql   # 或者: service mysql start
# 查看状态
systemctl status mysql  # 或者: service mysql status
```


## 命令行使用mysql

**登录数据库**

```bash
# 登录 MySQL，-u指定用户，-p提示输入密码
mysql -u root -p
```

**创建数据库与用户**

```sql
-- 创建数据库 testdb
CREATE DATABASE testdb;
-- 创建用户 testuser，设置密码为 password，'localhost' 表示只能从本机登录，使用 '%' 表示可以从任意主机登录
CREATE USER 'testuser'@'localhost' IDENTIFIED BY 'password';
-- 授权 testuser 对 testdb 拥有所有权限，此处的 'localhost' 同理
GRANT ALL PRIVILEGES ON testdb.* TO 'testuser'@'localhost';
-- 刷新权限
FLUSH PRIVILEGES;
-- 进入数据库
USE testdb;
-- 创建表
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INT
);
-- 插入数据
INSERT INTO users (name, age) VALUES ('Alice', 25);
```


**常用命令**

```sql
-- 显示当前数据库
SELECT DATABASE();
-- 列出所有数据库
SHOW DATABASES;
-- 使用某个数据库
USE <database_name>;
-- 查看所有用户
SELECT user, host FROM mysql.user;
-- 查看所有表
SHOW TABLES;
-- 查看表结构
DESCRIBE <table_name>;
-- 查看创建表的语句
SHOW CREATE TABLE <table_name>;
-- 退出 MySQL
exit;
```

## 常用sql语句

这部分在[postgresql_usage.md](./postgresql_usage.md#常用sql语句)里面已经介绍过了

## python使用mysql

```python
# 前提是在postgre中创建了testdb数据库跟testuser用户，并且在其中建立了users表格
# 并且授权testdb给testuser，并且给了权限
# 安装依赖: pip install mysql-connector-python

import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="testuser",
    password="password",
    database="testdb"
)

cur = conn.cursor()
cur.execute("SELECT * FROM users;")
rows = cur.fetchall()
for row in rows:
    print(row)

cur.close()
conn.close()
```