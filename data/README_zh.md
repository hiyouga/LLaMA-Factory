如果您使用自定义数据集，请务必在 `dataset_info.json` 文件中按照以下格式提供数据集定义。

```json
"数据集名称": {
  "hf_hub_url": "Hugging Face 的数据集仓库地址（若指定，则忽略 script_url 和 file_name）",
  "ms_hub_url": "ModelScope 的数据集仓库地址（若指定，则忽略 script_url 和 file_name）",
  "script_url": "包含数据加载脚本的本地文件夹名称（若指定，则忽略 file_name）",
  "file_name": "该目录下数据集文件的名称（若上述参数未指定，则此项必需）",
  "file_sha1": "数据集文件的 SHA-1 哈希值（可选，留空不影响训练）",
  "subset": "数据集子集的名称（可选，默认：None）",
  "folder": "Hugging Face 仓库的文件夹名称（可选，默认：None）",
  "ranking": "是否为偏好数据集（可选，默认：False）",
  "formatting": "数据集格式（可选，默认：alpaca，可以为 alpaca 或 sharegpt）",
  "columns（可选）": {
    "prompt": "数据集代表提示词的表头名称（默认：instruction）",
    "query": "数据集代表请求的表头名称（默认：input）",
    "response": "数据集代表回答的表头名称（默认：output）",
    "history": "数据集代表历史对话的表头名称（默认：None）",
    "messages": "数据集代表消息列表的表头名称（默认：conversations）",
    "system": "数据集代表系统提示的表头名称（默认：None）",
    "tools": "数据集代表工具描述的表头名称（默认：None）"
  },
  "tags（可选，用于 sharegpt 格式）": {
    "role_tag": "消息中代表发送者身份的键名（默认：from）",
    "content_tag": "消息中代表文本内容的键名（默认：value）",
    "user_tag": "消息中代表用户的 role_tag（默认：human）",
    "assistant_tag": "消息中代表助手的 role_tag（默认：gpt）",
    "observation_tag": "消息中代表工具返回结果的 role_tag（默认：observation）",
    "function_tag": "消息中代表工具调用的 role_tag（默认：function_call）",
    "system_tag": "消息中代表系统提示的 role_tag（默认：system，会覆盖 system 列）"
  }
}
```

添加后可通过指定 `--dataset 数据集名称` 参数使用自定义数据集。

----

该项目目前支持两种格式的数据集：**alpaca** 和 **sharegpt**，其中 alpaca 格式的数据集按照以下方式组织：

```json
[
  {
    "instruction": "用户指令（必填）",
    "input": "用户输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
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
    "system": "system",
    "history": "history"
  }
}
```

其中 `query` 列对应的内容会与 `prompt` 列对应的内容拼接后作为用户指令，即用户指令为 `prompt\nquery`。`response` 列对应的内容为模型回答。

`system` 列对应的内容将被作为系统提示词。`history` 列是由多个字符串二元组构成的列表，分别代表历史消息中每轮的指令和回答。注意历史消息中的回答**也会被用于训练**。

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

添加偏好数据集需要额外指定 `"ranking": true`。

----

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
    ],
    "system": "系统提示词（选填）",
    "tools": "工具描述（选填）"
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的 `columns` 应为：

```json
"数据集名称": {
  "columns": {
    "messages": "conversations",
    "system": "system",
    "tools": "tools"
  },
  "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt"
  }
}
```

其中 `messages` 列应当是一个列表，且符合 `用户/模型/用户/模型/用户/模型` 的顺序。

预训练数据集和偏好数据集尚不支持 sharegpt 格式。
