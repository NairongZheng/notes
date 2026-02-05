- [JSON Schema](#json-schema)
- [OpenAI 的 Function Calling](#openai-的-function-calling)
  - [Function 定义](#function-定义)
  - [Function Calling 请求与返回](#function-calling-请求与返回)


# JSON Schema

JSON Schema 官方链接：[https://json-schema.org/learn/getting-started-step-by-step](https://json-schema.org/learn/getting-started-step-by-step)（非常值得一看）


以下是官网的例子。

**最简单的 schema**

最简单的 schema 就是空：

```shell
{}
```

**创建一个 schema 定义**

| 字段        | 分类               | 是否影响校验 | 给谁看的      | 说明                                                                     |
| ----------- | ------------------ | ------------ | ------------- | ------------------------------------------------------------------------ |
| $schema     | schema keyword     | ❌            | Schema 引擎   | 告诉解析器：这个 Schema 使用的是哪一版 JSON Schema 规范                  |
| $id         | schema keyword     | ❌            | Schema 引擎   | 给当前 Schema 一个全局唯一标识                                           |
| title       | annotation         | ❌            | 人 / UI / LLM | 纯说明性字段                                                             |
| description | annotation         | ❌            | 人 / UI / LLM | 纯说明性字段                                                             |
| type        | validation keyword | ✅            | 校验器        | 真正的校验规则，**可选**：`object, array, string, number, boolean, null` |

其中 type 的选项：
- `object`: JSON 对象。`{"a": 1}`
- `array`: JSON 数组。`[1, 2, 3]`
- `string`: JSON 字符串。`"hello"`
- `number`: JSON 数字（整数+小数）。`1`, `3.14`
- `integer`: JSON 整数（number 子集）。`1`, `42`
- `boolean`: JSON 布尔值。`true`, `false`
- `null`: JSON 空值。`null`

```shell
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product in the catalog",
  "type": "object"
}
```

**定义 properties**

```shell
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    }
  }
}
```

**定义 required properties**

其中有几个范围限制验证参数：
- `x ≥ minimum`
- `x > exclusiveMinimum`
- `x ≤ maximum`
- `x < exclusiveMaximum`

其中的 `reqiured` 字段说明了必须的参数。

```shell
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    },
    "price": {
      "description": "The price of the product",
      "type": "number",
      "exclusiveMinimum": 0
    }
  },
  "required": [ "productId", "productName", "price" ]
}
```

**定义 opeional properties**

其中：
- `items`: 该例子中代表每个 `array` 里面的 `item` 是 `string` 类型
- `minItems`: 表明最少需要的 `item` 个数
- `uniqueItems`: 用来约束每个 `item` 是否唯一

```shell
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    },
    "price": {
      "description": "The price of the product",
      "type": "number",
      "exclusiveMinimum": 0
    },
    "tags": {
      "description": "Tags for the product",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": true
    }
  },
  "required": [ "productId", "productName", "price" ]
}
```

**创建嵌套的 Schema**

JSON Schema 是可以嵌套的，如下：

```shell
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    },
    "price": {
      "description": "The price of the product",
      "type": "number",
      "exclusiveMinimum": 0
    },
    "tags": {
      "description": "Tags for the product",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": true
    },
    "dimensions": {
      "type": "object",
      "properties": {
        "length": {
          "type": "number"
        },
        "width": {
          "type": "number"
        },
        "height": {
          "type": "number"
        }
      },
      "required": [ "length", "width", "height" ]
    }
  },
  "required": [ "productId", "productName", "price" ]
}
```

**引用 Schema**

JSON Schema 也可以引用外部的 Schema

假设现在有一个 Schema 如下：

```shell
{
  "$id": "https://example.com/geographical-location.schema.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Longitude and Latitude",
  "description": "A geographical coordinate on a planet (most commonly Earth).",
  "required": [ "latitude", "longitude" ],
  "type": "object",
  "properties": {
    "latitude": {
      "type": "number",
      "minimum": -90,
      "maximum": 90
    },
    "longitude": {
      "type": "number",
      "minimum": -180,
      "maximum": 180
    }
  }
}
```

那么可以使用 `$ref` 引用他：

```shell
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Product",
  "description": "A product from Acme's catalog",
  "type": "object",
  "properties": {
    "productId": {
      "description": "The unique identifier for a product",
      "type": "integer"
    },
    "productName": {
      "description": "Name of the product",
      "type": "string"
    },
    "price": {
      "description": "The price of the product",
      "type": "number",
      "exclusiveMinimum": 0
    },
    "tags": {
      "description": "Tags for the product",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": true
    },
    "dimensions": {
      "type": "object",
      "properties": {
        "length": {
          "type": "number"
        },
        "width": {
          "type": "number"
        },
        "height": {
          "type": "number"
        }
      },
      "required": [ "length", "width", "height" ]
    },
    "warehouseLocation": {
      "description": "Coordinates of the warehouse where the product is located.",
      "$ref": "https://example.com/geographical-location.schema.json"
    }
  },
  "required": [ "productId", "productName", "price" ]
}
```


# OpenAI 的 Function Calling

现在的 Function Calling 已经是属于 tool use 的一个子集了。

- OpenAI 的 Function Calling 官方链接：[https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
- OpenAI 的 tool use 官方链接：[https://platform.openai.com/docs/guides/tools](https://platform.openai.com/docs/guides/tools)

## Function 定义

Function 需要有以下 Properties：


| 字段          | 说明                                                                     |
| ------------- | ------------------------------------------------------------------------ |
| `type`        | 只能是 `function`                                                        |
| `name`        | Function 的名字                                                          |
| `description` | Function 的详细描述（非常重要，llm 主要就是通过这个描述来理解 Function） |
| `parameters`  | Function 的参数（使用 [JSON Schema 格式](#json-schema)）                 |
| `strict`      | Whether to enforce strict mode for the function call                     |

比如 `get_weather` 定义如下：

```shell
{
    "type": "function",
    "name": "get_weather",
    "description": "Retrieves current weather for the given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Units the temperature will be returned in."
            }
        },
        "required": ["location", "units"],
        "additionalProperties": false
    },
    "strict": true
}
```

## Function Calling 请求与返回

**请求 llm**

请求类似如下，其中的 tools 就是可选的工具列表：

```shell
response = client.responses.create(
    model="gpt-4.1",
    input=input_messages,
    tools=tools,
    tool_choice="auto",
)
```

其中 `tool choice` 参数来说明工具调用时，应该怎么选择。有以下几种选项：
- `Auto`: (Default) 可以调用 0, 1, 多个 Function。tool_choice: "auto"
- `Requited`: 至少调用一个 Function。tool_choice: "required"
- `Forced Function`: 强制调用某个指定的 Function。tool_choice: {"type": "function", "name": "get_weather"}
- `Allowed tools`: tool 白名单。只能从这里面选，也不是整个 tools 都能调用。

**llm 返回**

一般 llm 返回的工具调用格式可以有 0 个、1 个或者多个。可以参考（只是个例子，不要跟上面的工具定义对应，不然就会发现缺失了 required 参数）：

```shell
[
    {
        "id": "fc_12345xyz",
        "call_id": "call_12345xyz",
        "type": "function_call",
        "name": "get_weather",
        "arguments": "{\"location\":\"Paris, France\"}"
    },
    {
        "id": "fc_67890abc",
        "call_id": "call_67890abc",
        "type": "function_call",
        "name": "get_weather",
        "arguments": "{\"location\":\"Bogotá, Colombia\"}"
    },
    {
        "id": "fc_99999def",
        "call_id": "call_99999def",
        "type": "function_call",
        "name": "send_email",
        "arguments": "{\"to\":\"bob@email.com\",\"body\":\"Hi bob\"}"
    }
]
```

其中：
- `id`: 本次工具调用的唯一标识
- `call_id`: 用于把“模型发起的调用”和“返回的执行结果”对应起来的标识
- `name`: 调用的 Function 的名字
- `arguments`: 调用 Function 的参数

