如果您使用自定义数据集，请务必在 `dataset_info.json` 文件中按照以下格式提供数据集定义。

```json
"数据集名称": {
  "hf_hub_url": "Hugging Face 上的项目地址（若指定，则忽略下列三个参数）",
  "script_url": "包含数据加载脚本的本地文件夹名称（若指定，则忽略下列两个参数）",
  "file_name": "该目录下数据集文件的名称（若上述参数未指定，则此项必需）",
  "file_sha1": "数据集文件的SHA-1哈希值（可选，留空不影响训练）",
  "subset": "数据集子集的名称（可选，默认：None）",
  "ranking": "是否为偏好数据集（可选，默认：False）",
  "formatting": "数据集格式（可选，默认：alpaca，可以为 alpaca 或 sharegpt）",
  "columns": {
    "prompt": "数据集代表提示词的表头名称（默认：instruction，用于 alpaca 格式）",
    "query": "数据集代表请求的表头名称（默认：input，用于 alpaca 格式）",
    "response": "数据集代表回答的表头名称（默认：output，用于 alpaca 格式）",
    "history": "数据集代表历史对话的表头名称（默认：None，用于 alpaca 格式）",
    "messages": "数据集代表消息列表的表头名称（默认：conversations，用于 sharegpt 格式）",
    "role": "消息中代表发送者身份的键名（默认：from，用于 sharegpt 格式）",
    "content": "消息中代表文本内容的键名（默认：value，用于 sharegpt 格式）"
  }
}
```

添加后可通过指定 `--dataset 数据集名称` 参数使用自定义数据集。

该项目目前支持两种格式的数据集：**alpaca** 和 **sharegpt**，其中 alpaca 格式的数据集按照以下方式组织：

```json
[
  {
    "instruction": "用户指令（必填）",
    "input": "用户输入（选填）",
    "output": "模型回答（必填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的 `columns` 应为：

```json
"数据集名称": {
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "history": "history"
  }
}
```

其中 `prompt` 和 `response` 列应当是非空的字符串，分别代表用户指令和模型回答。`query` 列的内容将会和 `prompt` 列拼接作为模型输入。

`history` 列是由多个字符串二元组构成的列表，分别代表历史消息中每轮的指令和回答。注意每轮的模型回答**均会被用于训练**。

对于预训练数据集，仅 `prompt` 列中的内容会用于模型训练。

对于偏好数据集，`response` 列应当是一个长度为 2 的字符串列表，排在前面的代表更优的回答，例如：

```json
{
  "instruction": "用户指令",
  "input": "用户输入",
  "output": [
    "优质回答",
    "劣质回答"
  ]
}
```

而 sharegpt 格式的数据集按照以下方式组织：

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "用户指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的 `columns` 应为：

```json
"数据集名称": {
  "columns": {
    "messages": "conversations",
    "role": "from",
    "content": "value"
  }
}
```

其中 `messages` 列必须为偶数长度的列表，且符合 `用户/模型/用户/模型/用户/模型` 的顺序。

预训练数据集和偏好数据集尚不支持 sharegpt 格式。
