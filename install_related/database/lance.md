# Lance：基础概念与使用

- [Lance 是什么](#lance-是什么)
- [行式与列式](#行式与列式)
- [可空字段与空值](#可空字段与空值)
- [最小操作](#最小操作)

## Lance 是什么

JSONL 是每行一个 JSON 值的文本格式；训练数据通常每行放一条样本。Lance 是面向 AI/多模态数据的开源列式存储格式，也定义了管理数据集版本等信息的表格式。入门时可把它理解成“用 Python 打开的一张有类型、有版本的持久化表”。

三个名字容易混淆：

| 名字 | 作用 |
| --- | --- |
| Lance | 底层文件与表格式，负责保存和组织数据 |
| pylance | 操作 Lance 的 Python 包，代码中使用 `import lance` |
| LanceDB | 构建在 Lance 之上的数据库产品；使用 Lance 基础读写 API 不需要先部署它 |

Arrow 是内存中的列式数据表示，PyArrow 是它的 Python 工具库。Arrow Table 可以理解成一张有字段类型的内存表；写入 Lance 后，数据就持久化到了存储中。

参考：[Lance 入门](https://lance.org/quickstart/)、[LanceDB](https://www.lancedb.com/)。资料核对日期：2026-10-15。

## 行式与列式

同样的三条样本：

```jsonl
{"id":1,"domain":"math","text":"求 1+1"}
{"id":2,"domain":"code","text":"写排序"}
{"id":3,"domain":"math","text":"解方程"}
```

JSONL 把一条样本的字段写在同一行。普通逐行扫描即使只需要 `domain`，也仍需经过包含 `text` 的整行文本。

列式存储把同一字段的数据按列组织。只读取 `domain` 时，可以跳过无关的正文列。这称为**列裁剪（projection）**。下面只是逻辑示意，不表示每列各占一个文件：

```text
id      → [1, 2, 3]
domain  → [math, code, math]
text    → [求 1+1, 写排序, 解方程]
```

选择字段和筛选样本是两回事：

- `columns=["id"]`：返回哪些字段。
- `filter="domain = 'math'"`：保留哪些样本；即使不返回 `domain`，也需要它来判断条件。
- `take([0, 2])`：按当前位置读取第 1、3 行。位置从 0 开始，和业务字段 `id` 不是一回事。

筛选不等于自动建索引；没有合适索引时仍可能扫描条件列。实际速度取决于数据大小、访问方式、压缩、索引和存储介质，不能从三条样本推导性能倍数。

参考：[Lance 读写指南](https://lance.org/guide/read_and_write/)。

## 可空字段与空值

Lance 允许字段为 `null`，schema 中的 `nullable` 表示该字段是否允许空值。schema 是整张表的字段定义；某一行没有值，不等于整张表没有这个字段。Python 中用 `None` 表示这里的空值，读回 Python 时也表现为 `None`。

在原来的三条样本上增加 `score`（评分）、`note`（备注）和 `tags`（标签）：

| id | score | note | tags |
| --- | --- | --- | --- |
| 1 | 0.95 | `"已复核"` | `["简单"]` |
| 2 | `None` | `""` | `[]` |
| 3 | 未提供这个键 | `None` | `None` |

这些状态有不同含义，业务上应约定清楚：

- `score=None`：没有评分；`score=0.0` 则是有评分且评分为零。
- `note=None`：没有备注值；`note=""` 是一个存在但长度为零的字符串。
- `tags=None`：没有标签列表值，例如尚未标注；`tags=[]` 是存在但没有元素的列表，例如已标注且没有标签。
- 在本例的显式 schema 下，`pa.Table.from_pylist(..., schema=schema)` 将缺少的 `score` 键补为 null。写入后无法仅凭这一列区分“原来没写这个键”和“原来明确写了 None”。

空值不会让整行消失，也不妨碍读取这一行的其他字段。列类型仍保持一致，例如 `score` 是允许 null 的浮点数列，而不是任意 Python 对象。

显式指定类型也能避免“这批评分全部为 None，不知道它应是什么类型”的问题。`nullable=True` 表示某个具体类型允许缺失值，不等于字段类型本身是 `pa.null()`。PyArrow 的 `pa.field()` 默认允许 null；示例明确写出可选字段的设置，便于理解。

查询空值使用 `IS NULL` / `IS NOT NULL`，不要写 `= NULL`。`note = ''` 只匹配空字符串，不匹配 null；`score >= 0.9` 也不会选中空评分。

参考：[Lance Schema](https://lance.org/format/table/schema/)、[Arrow 数据模型](https://arrow.apache.org/docs/python/data.html)、[Lance 筛选表达式](https://lance.org/guide/read_and_write/#filter-push-down)。

## 最小操作

依赖为 `pylance` 与 `pyarrow`。以下为官方接口改写的教学示例，Lance 部分尚未经过本机运行验证。使用临时目录，不覆盖已有数据集。

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import lance
import pyarrow as pa

with TemporaryDirectory(prefix="lance-demo-") as work_dir:
    # 定义每列的类型；即使评分全为空，也保留浮点数类型。
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("domain", pa.string()),
        pa.field("text", pa.string()),
        pa.field("score", pa.float64(), nullable=True),
        pa.field("note", pa.string(), nullable=True),
        pa.field("tags", pa.list_(pa.string()), nullable=True),
    ])
    rows = [
        {"id": 1, "domain": "math", "text": "求 1+1",
         "score": 0.95, "note": "已复核", "tags": ["简单"]},
        {"id": 2, "domain": "code", "text": "写排序",
         "score": None, "note": "", "tags": []},
        # 没有提供 score 键：按上面的 schema 补为 null。
        {"id": 3, "domain": "math", "text": "解方程",
         "note": None, "tags": None},
    ]
    table = pa.Table.from_pylist(rows, schema=schema)
    uri = str(Path(work_dir) / "samples.lance")
    lance.write_dataset(table, uri)
    ds = lance.dataset(uri)

    # 查看字段类型与总行数。
    print(ds.schema)
    print(ds.count_rows())  # 预期：3

    # 筛选 math 样本，只返回 id；domain 用来判断条件。
    print(ds.to_table(columns=["id"], filter="domain = 'math'").to_pylist())
    # 预期：[{'id': 1}, {'id': 3}]

    # 按行位置读取两条样本，不是按业务 id 查询。
    print(ds.take([0, 2], columns=["text"]).to_pylist())
    # 预期：[{'text': '求 1+1'}, {'text': '解方程'}]

    # 显式 None 与缺少 score 键，都会被 IS NULL 选中。
    print(ds.to_table(columns=["id", "score"], filter="score IS NULL").to_pylist())
    # 预期：[{'id': 2, 'score': None}, {'id': 3, 'score': None}]

    # 空字符串与 null 是两个不同条件。
    print(ds.to_table(columns=["id"], filter="note = ''").to_pylist())
    # 预期：[{'id': 2}]
    print(ds.to_table(columns=["id"], filter="note IS NULL").to_pylist())
    # 预期：[{'id': 3}]

    # 查看空列表与 null，读回 Python 后仍能区分。
    print(ds.take([1, 2], columns=["tags"]).to_pylist())
    # 预期：[{'tags': []}, {'tags': None}]
```

`samples.lance` 在这个数据集 API 中是目录。典型内容包括 `data/` 下的数据文件与 `_versions/` 下的版本清单；后者记录某个版本使用哪些数据及其元信息。单个内部数据文件不等于完整的数据集。

这段示例退出 `with` 后会清理临时目录。实际保存时，应选择持久目录。大数据应分批读取和写入，避免一次性把整张表载入内存。

参考：[Lance 存储布局](https://lance.org/format/table/layout/)。
