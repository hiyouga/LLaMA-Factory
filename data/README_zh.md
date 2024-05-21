[dataset_info.json](dataset_info.json) 包含了所有可用的数据集。如果您希望使用自定义数据集，请**务必**在 `dataset_info.json` 文件中添加*数据集描述*，并通过修改 `dataset: 数据集名称` 配置来使用数据集。

目前我们支持 **alpaca** 格式和 **sharegpt** 格式的数据集。

```json
"数据集名称": {
  "hf_hub_url": "Hugging Face 的数据集仓库地址（若指定，则忽略 script_url 和 file_name）",
  "ms_hub_url": "ModelScope 的数据集仓库地址（若指定，则忽略 script_url 和 file_name）",
  "script_url": "包含数据加载脚本的本地文件夹名称（若指定，则忽略 file_name）",
  "file_name": "该目录下数据集文件夹或文件的名称（若上述参数未指定，则此项必需）",
  "formatting": "数据集格式（可选，默认：alpaca，可以为 alpaca 或 sharegpt）",
  "ranking": "是否为偏好数据集（可选，默认：False）",
  "subset": "数据集子集的名称（可选，默认：None）",
  "folder": "Hugging Face 仓库的文件夹名称（可选，默认：None）",
  "columns（可选）": {
    "prompt": "数据集代表提示词的表头名称（默认：instruction）",
    "query": "数据集代表请求的表头名称（默认：input）",
    "response": "数据集代表回答的表头名称（默认：output）",
    "history": "数据集代表历史对话的表头名称（默认：None）",
    "messages": "数据集代表消息列表的表头名称（默认：conversations）",
    "system": "数据集代表系统提示的表头名称（默认：None）",
    "tools": "数据集代表工具描述的表头名称（默认：None）",
    "images": "数据集代表图像输入的表头名称（默认：None）",
    "chosen": "数据集代表更优回答的表头名称（默认：None）",
    "rejected": "数据集代表更差回答的表头名称（默认：None）",
    "kto_tag": "数据集代表 KTO 标签的表头名称（默认：None）"
  },
  "tags（可选，用于 sharegpt 格式）": {
    "role_tag": "消息中代表发送者身份的键名（默认：from）",
    "content_tag": "消息中代表文本内容的键名（默认：value）",
    "user_tag": "消息中代表用户的 role_tag（默认：human）",
    "assistant_tag": "消息中代表助手的 role_tag（默认：gpt）",
    "observation_tag": "消息中代表工具返回结果的 role_tag（默认：observation）",
    "function_tag": "消息中代表工具调用的 role_tag（默认：function_call）",
    "system_tag": "消息中代表系统提示的 role_tag（默认：system，会覆盖 system column）"
  }
}
```

## Alpaca 格式

### 指令监督微调数据集

- [样例数据集](alpaca_zh_demo.json)

在指令监督微调时，`instruction` 列对应的内容会与 `input` 列对应的内容拼接后作为人类指令，即人类指令为 `instruction\ninput`。而 `output` 列对应的内容为模型回答。

如果指定，`system` 列对应的内容将被作为系统提示词。

`history` 列是由多个字符串二元组构成的列表，分别代表历史消息中每轮对话的指令和回答。注意在指令监督微调时，历史消息中的回答内容**也会被用于模型学习**。

```json
[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

### 预训练数据集

- [样例数据集](c4_demo.json)

在预训练时，只有 `text` 列中的内容会用于模型学习。

```json
[
  {"text": "document"},
  {"text": "document"}
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "text"
  }
}
```

### 偏好数据集

偏好数据集用于奖励模型训练、DPO 训练和 ORPO 训练。

它需要在 `chosen` 列中提供更优的回答，并在 `rejected` 列中提供更差的回答。

```json
[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "chosen": "优质回答（必填）",
    "rejected": "劣质回答（必填）"
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "ranking": true,
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### KTO 数据集

- [样例数据集](kto_en_demo.json)

KTO 数据集需要额外添加一个 `kto_tag` 列，包含 bool 类型的人类反馈。

```json
[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "kto_tag": "人类反馈 [true/false]（必填）"
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "kto_tag": "kto_tag"
  }
}
```

### 多模态数据集

- [样例数据集](mllm_demo.json)

多模态数据集需要额外添加一个 `images` 列，包含输入图像的路径。目前我们仅支持单张图像输入。

```json
[
  {
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "images": [
      "图像路径（必填）"
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "images": "images"
  }
}
```

## Sharegpt 格式

### 指令监督微调数据集

- [样例数据集](glaive_toolcall_zh_demo.json)

相比 alpaca 格式的数据集，sharegpt 格式支持**更多的角色种类**，例如 human、gpt、observation、function 等等。它们构成一个对象列表呈现在 `conversations` 列中。

注意其中 human 和 observation 必须出现在奇数位置，gpt 和 function 必须出现在偶数位置。

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "function_call",
        "value": "工具参数"
      },
      {
        "from": "observation",
        "value": "工具结果"
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

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system",
    "tools": "tools"
  }
}
```

### 偏好数据集

- [样例数据集](dpo_zh_demo.json)

Sharegpt 格式的偏好数据集同样需要在 `chosen` 列中提供更优的消息，并在 `rejected` 列中提供更差的消息。

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      },
      {
        "from": "human",
        "value": "人类指令"
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "优质回答"
    },
    "rejected": {
      "from": "gpt",
      "value": "劣质回答"
    }
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "ranking": true,
  "columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### OpenAI 格式

OpenAI 格式仅仅是 sharegpt 格式的一种特殊情况，其中第一条消息可能是系统提示词。

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "系统提示词（选填）"
      },
      {
        "role": "user",
        "content": "人类指令"
      },
      {
        "role": "assistant",
        "content": "模型回答"
      }
    ]
  }
]
```

对于上述格式的数据，`dataset_info.json` 中的*数据集描述*应为：

```json
"数据集名称": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system"
  }
}
```

Sharegpt 格式中的 KTO 数据集和多模态数据集与 alpaca 格式的类似。

预训练数据集**不支持** sharegpt 格式。
