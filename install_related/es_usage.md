- [基本概念快速理解](#基本概念快速理解)
- [安装elasticsearch](#安装elasticsearch)
- [安装kibana](#安装kibana)
- [使用步骤](#使用步骤)
- [常用查询语句](#常用查询语句)
- [python使用elasticsearch示例](#python使用elasticsearch示例)
- [可视化管理工具Kibana](#可视化管理工具kibana)


## 基本概念快速理解

| 概念             | 含义                                  |
| ---------------- | ------------------------------------- |
| Index（索引）    | 类似数据库中的表                      |
| Document（文档） | 类似表中的一行记录，JSON 格式         |
| Field（字段）    | 文档的每个属性                        |
| Mapping          | 类似表结构定义，规定字段类型          |
| Query DSL        | Elasticsearch 的查询语言（JSON 格式） |

## 安装elasticsearch

```bash
# docker启动单节点es
docker run -d --name elasticsearch -e "discovery.type=single-node" -e "xpack.security.enabled=false" -p 9200:9200 -p 9300:9300 docker.elastic.co/elasticsearch/elasticsearch:9.0.3
docker run -it --name elasticsearch -v /tmp/es_data:/usr/share/elasticsearch/data -p 9200:9200 -p 9300:9300 docker.elastic.co/elasticsearch/elasticsearch:9.0.3
# 9200: HTTP API 端口（用于访问）
# 9300: 节点间通信端口
# xpack.security.enabled=false: 禁用安全认证，方便本地测试
# 正常是需要做数据卷映射之类的，将容器的 /usr/share/elasticsearch/data 映射到数据盘
```

## 安装kibana

```bash
# 启动kibana（用来可视化的）
docker run --name kibana -p 5601:5601 docker.elastic.co/kibana/kibana:9.0.3
# 好像也可以用 apt install 的方式
```

## 使用步骤

**1、创建索引（可省略，写入时会自动创建）**
```bash
curl -X PUT http://localhost:9200/test_database
```

带 Mapping 创建（指定字段类型）：
```bash
curl -X PUT http://localhost:9200/test_database -H 'Content-Type: application/json' -d '{
  "mappings": {
    "properties": {
      "title": { "type": "text" },
      "views": { "type": "integer" },
      "timestamp": { "type": "date" }
    }
  }
}'
```
**2、插入文档（写入数据）**
```bash
curl -X POST http://localhost:9200/test_database/_doc/1 -H 'Content-Type: application/json' -d '{
  "title": "Elasticsearch 入门",
  "views": 100,
  "timestamp": "2025-07-23T12:00:00"
}'
```
可以省略 id，让 ES 自动生成：

```bash
curl -X POST http://localhost:9200/test_database/_doc -H 'Content-Type: application/json' -d '{
  "title": "用 Python 操作 Elasticsearch",
  "views": 42,
  "timestamp": "2025-07-23T15:30:00"
}'
```
**3、查询数据（全文检索）**

**3.1 查询所有文档**

```bash
curl -X GET http://localhost:9200/test_database/_search?pretty
```

**3.2 按条件搜索（match）**

```bash
curl -X GET http://localhost:9200/test_database/_search -H 'Content-Type: application/json' -d '{
  "query": {
    "match": {
      "title": "Elasticsearch"
    }
  }
}'
```

**3.3 精确匹配（term 查询）**

```bash
curl -X GET http://localhost:9200/test_database/_search -H 'Content-Type: application/json' -d '{
  "query": {
    "term": {
      "views": 100
    }
  }
}'
```

**3.4 范围查询（range）**

```bash
curl -X GET http://localhost:9200/test_database/_search -H 'Content-Type: application/json' -d '{
  "query": {
    "range": {
      "views": {
        "gte": 50,
        "lte": 200
      }
    }
  }
}'
```

## 常用查询语句

**索引操作**

```bash
# 创建索引
curl -X PUT "localhost:9200/test_database"
# 删除索引
curl -X DELETE "localhost:9200/test_database"
# 查看所有索引
curl -X GET "localhost:9200/_cat/indices?v"
```

**文档操作**

```bash
# 插入文档
curl -X POST "localhost:9200/test_database/_doc/1" -H 'Content-Type: application/json' -d '{"name": "Alice", "age": 25}'
# 获取文档
curl -X GET "localhost:9200/test_database/_doc/1"
# 更新文档
curl -X POST "localhost:9200/test_database/_update/1" -H 'Content-Type: application/json' -d '{"doc": {"age": 26}}'
# 删除文档
curl -X DELETE "localhost:9200/test_database/_doc/1"
```

**查询操作**

```bash
# 查询所有文档
curl -X GET "localhost:9200/test_database/_search" -H 'Content-Type: application/json' -d '{"query": {"match_all": {}}}'

# 条件查询
curl -X GET "localhost:9200/test_database/_search" -H 'Content-Type: application/json' -d '{"query": {"match": {"name": "Alice"}}}'

# 范围查询
curl -X GET "localhost:9200/test_database/_search" -H 'Content-Type: application/json' -d '{"query": {"range": {"age": {"gte": 20, "lte": 30}}}}'

# 分页、排序
curl -X GET "localhost:9200/test_database/_search" -H 'Content-Type: application/json' -d '{"from": 0, "size": 10, "sort": [{"age": "desc"}]}'
```

## python使用elasticsearch示例

```python
# 需要先安装elasticsearch库: pip install elasticsearch
from elasticsearch import Elasticsearch

es = Elasticsearch(["http://localhost:9200"])

# 插入文档
doc = {"name": "Alice", "age": 25}
es.index(index="test_database", id=1, document=doc)

# 查询文档
res = es.get(index="test_database", id=1)
print(res["_source"])

# 搜索
query = {"query": {"match": {"name": "Alice"}}}
res = es.search(index="test_database", body=query)
for hit in res["hits"]["hits"]:
    print(hit["_source"])

# 更新文档
es.update(index="test_database", id=1, body={"doc": {"age": 26}})

# 删除文档
es.delete(index="test_database", id=1)
```

## 可视化管理工具Kibana

Kibana是Elastic官方的可视化管理工具。

**安装Kibana**

```bash
apt install kibana
systemctl start kibana
systemctl enable kibana
```

**访问Kibana**

默认监听5601端口，浏览器访问：http://localhost:5601

**常用功能**
- 可视化数据分析
- 管理索引、数据
- Dev Tools（可直接执行ES语句）

