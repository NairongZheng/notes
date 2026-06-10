- [关于 S3 与本工具的关系](#关于-s3-与本工具的关系)
- [安装与配置](#安装与配置)
- [基本概念](#基本概念)
- [bucket 操作](#bucket-操作)
- [对象操作](#对象操作)
- [同步与批量](#同步与批量)
- [预签名 URL](#预签名-url)
- [分片上传](#分片上传)
- [存储类与生命周期](#存储类与生命周期)
- [权限与策略](#权限与策略)
- [常用参数](#常用参数)
- [python 使用 s3 示例（boto3）](#python-使用-s3-示例boto3)


# 关于 S3 与本工具的关系

S3 有两层含义：
- **狭义**：AWS 的 Amazon Simple Storage Service 服务
- **广义**：对象存储 api 规范。市面上对象存储几乎都兼容（minio / 阿里 oss / cloudflare r2 / 七牛 / 腾讯 cos 等）

操作 s3 的常见工具：

| 工具      | 范围                       | 用法               |
| --------- | -------------------------- | ------------------ |
| `aws cli` | AWS 全家桶（含 s3）        | 日常运维、写脚本   |
| `boto3`   | AWS 全家桶 python sdk      | 业务代码集成       |
| `rclone`  | 70+ 存储后端（s3 是其一）  | 跨云搬数据         |

> 本文档聚焦 **AWS S3 + aws cli + boto3**。命令大多对其他 s3 兼容存储也适用，加 `--endpoint-url` 即可。
> 跨云/多后端同步见 `tools/rclone_usage.md`。


# 安装与配置

```shell
# 安装 aws cli v2 (linux x86_64)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws --version

# 交互式配置 aksk，会写入 ~/.aws/credentials 和 ~/.aws/config
aws configure
    # AWS Access Key ID:     xxxx
    # AWS Secret Access Key: xxxx
    # Default region name:   ap-southeast-1
    # Default output format: json

# 配置多个 profile（多账号必备）
aws configure --profile myprofile
# 使用：aws s3 ls --profile myprofile  或  export AWS_PROFILE=myprofile
```

**配置文件结构**

```shell
# ~/.aws/credentials                    # ~/.aws/config
[default]                               [default]
aws_access_key_id = xxxx                region = ap-southeast-1
aws_secret_access_key = xxxx            output = json

[myprofile]                             [profile myprofile]
aws_access_key_id = xxxx                region = us-east-1
aws_secret_access_key = xxxx
```

# 基本概念

- **bucket**：桶，名字全局唯一（所有 AWS 账户共享命名空间），属于某个 region
- **object**：对象（文件），由 key（路径）和 value（内容）组成
- **key / prefix**：s3 没有真正的"目录"，目录只是 key 的前缀，比如 `images/2026/a.png` 的前缀是 `images/`
- **storage class**：存储类，决定价格和访问速度（Standard / IA / Glacier 等）
- **s3 vs s3api**：aws cli 操作 s3 有两套子命令：
    - `aws s3`：高级命令，类 unix 风格，7 个动词（ls/cp/mv/rm/sync/mb/rb），日常传文件用它
    - `aws s3api`：底层 api 直通车，bucket policy / 生命周期 / 版本控制 / 分片 / metadata 都靠它
    - 口诀：**传文件用 s3，改配置用 s3api**

# bucket 操作

```shell
# 列出所有 bucket
aws s3 ls
# 创建 bucket（名字全局唯一）
aws s3 mb s3://my-bucket --region ap-southeast-1
# 删除 bucket（空 / 强制）
aws s3 rb s3://my-bucket
aws s3 rb s3://my-bucket --force
# 查看 bucket 所在 region
aws s3api get-bucket-location --bucket my-bucket
# 查看 bucket 总大小
aws s3 ls s3://my-bucket --recursive --summarize --human-readable | tail -2
```

# 对象操作

```shell
# 列出文件
aws s3 ls s3://my-bucket/                                    # 当前层
aws s3 ls s3://my-bucket/ --recursive --human-readable       # 递归 + 易读

# 上传 / 下载（cp 自动判断方向）
aws s3 cp ./local.txt s3://my-bucket/path/                   # 上传
aws s3 cp ./local_dir s3://my-bucket/path/ --recursive       # 上传目录
aws s3 cp s3://my-bucket/path/file.txt ./                    # 下载
aws s3 cp s3://my-bucket/path/ ./local_dir --recursive       # 下载目录
aws s3 cp ./big.zip s3://my-bucket/ --storage-class STANDARD_IA  # 指定存储类

# 移动 / 删除
aws s3 mv s3://bucket-a/a.txt s3://bucket-b/a.txt            # 跨 bucket 移动
aws s3 rm s3://my-bucket/path/file.txt                       # 删单个
aws s3 rm s3://my-bucket/path/ --recursive                   # 递归删（小心！）

# 查看对象元信息（不下载内容）
aws s3api head-object --bucket my-bucket --key path/file.txt
```

# 同步与批量

```shell
# 本地 -> s3（增量，不删除目标多余文件）
aws s3 sync ./local_dir s3://my-bucket/path/
# s3 -> 本地
aws s3 sync s3://my-bucket/path/ ./local_dir
# 严格对齐（删除目标端多余文件，小心！）
aws s3 sync ./local_dir s3://my-bucket/path/ --delete
# 过滤（顺序敏感，后面的覆盖前面的）
aws s3 sync ./ s3://my-bucket/ --exclude "*" --include "*.jpg"
# 模拟，不实际执行
aws s3 sync ./local_dir s3://my-bucket/path/ --dryrun
```

# 预签名 URL

> 给没有 aksk 的人临时访问对象，常用于前端下载/上传链接。

```shell
# 下载链接（GET），默认 3600 秒，最长 7 天
aws s3 presign s3://my-bucket/path/file.txt --expires-in 86400
```

> 上传用的预签名（PUT）cli 不直接支持，用 boto3 生成（见末尾 boto3 章节）。

# 分片上传

> aws s3 cp 上传大文件时会**自动分片**，日常不用手动管。
> s3api 提供手动分片 api（create-multipart-upload / upload-part / complete-multipart-upload），需要时查文档。

```shell
# 重要：清理未完成的分片上传，否则会持续占用存储费用
aws s3api list-multipart-uploads --bucket my-bucket
aws s3api abort-multipart-upload --bucket my-bucket --key big.zip --upload-id xxxx
```

> 推荐配合生命周期规则自动清理（见下一节，`AbortIncompleteMultipartUpload`）。

# 存储类与生命周期

| 存储类               | 用途                     | 取回时间        |
| -------------------- | ------------------------ | --------------- |
| STANDARD             | 默认，频繁访问           | 毫秒            |
| STANDARD_IA          | 不常访问（最少 30 天）   | 毫秒            |
| ONEZONE_IA           | 同上，单可用区，更便宜   | 毫秒            |
| INTELLIGENT_TIERING  | 自动分层                 | 毫秒            |
| GLACIER_IR           | 归档但快速访问           | 毫秒            |
| GLACIER              | 归档                     | 分钟 ~ 小时     |
| DEEP_ARCHIVE         | 最便宜，长期归档         | 小时（最长12h）|

```shell
# 从 Glacier 恢复对象（取回 3 天）
aws s3api restore-object --bucket my-bucket --key big.zip \
    --restore-request '{"Days":3,"GlacierJobParameters":{"Tier":"Standard"}}'
```

**生命周期规则**（自动转存储类/过期/清理分片）

```shell
cat > lifecycle.json <<'EOF'
{
  "Rules": [{
    "ID": "log-archive",
    "Status": "Enabled",
    "Filter": {"Prefix": "logs/"},
    "Transitions": [
      {"Days": 30, "StorageClass": "STANDARD_IA"},
      {"Days": 90, "StorageClass": "GLACIER"}
    ],
    "Expiration": {"Days": 365},
    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}
  }]
}
EOF

aws s3api put-bucket-lifecycle-configuration --bucket my-bucket \
    --lifecycle-configuration file://lifecycle.json
aws s3api get-bucket-lifecycle-configuration --bucket my-bucket
```

# 权限与策略

**bucket policy**（json，控制谁可以做什么）

```shell
# 查看 / 删除
aws s3api get-bucket-policy --bucket my-bucket
aws s3api delete-bucket-policy --bucket my-bucket

# 设置（示例：允许公开只读）
cat > policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow", "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::my-bucket/*"
  }]
}
EOF
aws s3api put-bucket-policy --bucket my-bucket --policy file://policy.json
```

**block public access**（防止误开放公开访问）

```shell
aws s3api get-public-access-block --bucket my-bucket
# 全部禁用公开（推荐默认）
aws s3api put-public-access-block --bucket my-bucket \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

**版本控制**（开启后不可关闭，只能 Suspended）

```shell
aws s3api put-bucket-versioning --bucket my-bucket \
    --versioning-configuration Status=Enabled
aws s3api list-object-versions --bucket my-bucket   # 列出所有版本
```

# 常用参数

| 参数                       | 功能                               | 说明                       |
| -------------------------- | ---------------------------------- | -------------------------- |
| `--profile NAME`           | 指定 profile                       | 多账号场景                 |
| `--region REGION`          | 指定 region                        | 跨区操作                   |
| `--recursive`              | 递归                               | cp / rm / ls 常用          |
| `--dryrun`                 | 模拟执行不动                       | sync / rm 前先跑一遍       |
| `--exclude` / `--include`  | 过滤文件                           | 顺序敏感，后面覆盖前面     |
| `--storage-class CLASS`    | 指定存储类                         | 上传时                     |
| `--sse`                    | 服务端加密（AES256 / aws:kms）     | 合规要求时                 |
| `--delete`                 | sync 时删除目标多余文件            | 小心！                     |
| `--endpoint-url URL`       | 自定义 endpoint                    | 用 minio / oss 等兼容存储 |

# python 使用 s3 示例（boto3）

> boto3 是 AWS 官方 python sdk，日常用 `client` 接口即可。

```shell
pip install boto3
pip install 'boto3-stubs[s3]'   # 可选，类型提示
```

**初始化**

```python
import boto3

# 用默认 profile
s3 = boto3.client("s3")

# 指定 profile
s3 = boto3.Session(profile_name="myprofile").client("s3")

# 显式传 aksk（不推荐写死在代码）
s3 = boto3.client(
    "s3",
    aws_access_key_id="xxxx",
    aws_secret_access_key="xxxx",
    region_name="ap-southeast-1",
    # endpoint_url="https://minio.example.com",   # 兼容 s3 协议的存储
)
```

**列表（注意单次最多 1000 个，超过要分页）**

```python
# 列 bucket
for b in s3.list_buckets()["Buckets"]:
    print(b["Name"])

# 列对象（分页器，处理大量对象）
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket="my-bucket", Prefix="logs/"):
    for obj in page.get("Contents", []):
        print(obj["Key"], obj["Size"])
```

**上传 / 下载**

```python
# 文件上传（大文件自动分片）
s3.upload_file("local.txt", "my-bucket", "path/remote.txt")

# 带额外参数
s3.upload_file("big.zip", "my-bucket", "path/big.zip", ExtraArgs={
    "StorageClass": "STANDARD_IA",
    "ContentType": "application/zip",
    "Metadata": {"author": "damon"},
})

# 内存 bytes 上传 / 读取
s3.put_object(Bucket="my-bucket", Key="data.bin", Body=b"hello")
content = s3.get_object(Bucket="my-bucket", Key="data.bin")["Body"].read()

# 下载文件
s3.download_file("my-bucket", "path/remote.txt", "local.txt")
```

**删除 / 判断存在**

```python
# 单个 / 批量（单次最多 1000 个）
s3.delete_object(Bucket="my-bucket", Key="path/file.txt")
s3.delete_objects(Bucket="my-bucket",
    Delete={"Objects": [{"Key": "a.txt"}, {"Key": "b.txt"}]})

# head_object 判断存在（不下载内容）
from botocore.exceptions import ClientError
def s3_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise
```

**复制 / 删除前缀下所有对象**

```python
# 复制（同 / 跨 bucket 都行）
s3.copy_object(Bucket="dst", Key="path/file.txt",
    CopySource={"Bucket": "src", "Key": "path/file.txt"})

# s3 没有"删目录"，要自己遍历
def delete_prefix(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": objs})
```

**预签名 URL（GET 下载 / PUT 上传）**

```python
# 下载链接
url = s3.generate_presigned_url("get_object",
    Params={"Bucket": "my-bucket", "Key": "path/file.txt"},
    ExpiresIn=3600)   # 秒，最长 7 天

# 上传链接（前端可直接 PUT 文件上来）
url = s3.generate_presigned_url("put_object",
    Params={"Bucket": "my-bucket", "Key": "path/upload.txt"},
    ExpiresIn=3600)
```

**进度条**

```python
# upload_file / download_file 都支持 Callback
from tqdm import tqdm
import os

size = os.path.getsize("big.zip")
with tqdm(total=size, unit="B", unit_scale=True, desc="upload") as bar:
    s3.upload_file("big.zip", "my-bucket", "big.zip",
        Callback=lambda n: bar.update(n))
```
