The [dataset_info.json](dataset_info.json) contains all available datasets. If you are using a custom dataset, please **make sure** to add a *dataset description* in `dataset_info.json` and specify `dataset: dataset_name` before training to use it.

Currently we support datasets in **alpaca** and **sharegpt** format.

```json
"dataset_name": {
  "hf_hub_url": "the name of the dataset repository on the Hugging Face hub. (if specified, ignore script_url and file_name)",
  "ms_hub_url": "the name of the dataset repository on the Model Scope hub. (if specified, ignore script_url and file_name)",
  "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore file_name)",
  "file_name": "the name of the dataset folder or dataset file in this directory. (required if above are not specified)",
  "formatting": "the format of the dataset. (optional, default: alpaca, can be chosen from {alpaca, sharegpt})",
  "ranking": "whether the dataset is a preference dataset or not. (default: False)",
  "subset": "the name of the subset. (optional, default: None)",
  "folder": "the name of the folder of the dataset repository on the Hugging Face hub. (optional, default: None)",
  "num_samples": "the number of samples in the dataset used for training. (optional, default: None)",
  "columns (optional)": {
    "prompt": "the column name in the dataset containing the prompts. (default: instruction)",
    "query": "the column name in the dataset containing the queries. (default: input)",
    "response": "the column name in the dataset containing the responses. (default: output)",
    "history": "the column name in the dataset containing the histories. (default: None)",
    "messages": "the column name in the dataset containing the messages. (default: conversations)",
    "system": "the column name in the dataset containing the system prompts. (default: None)",
    "tools": "the column name in the dataset containing the tool description. (default: None)",
    "images": "the column name in the dataset containing the image inputs. (default: None)",
    "chosen": "the column name in the dataset containing the chosen answers. (default: None)",
    "rejected": "the column name in the dataset containing the rejected answers. (default: None)",
    "kto_tag": "the column name in the dataset containing the kto tags. (default: None)"
  },
  "tags (optional, used for the sharegpt format)": {
    "role_tag": "the key in the message represents the identity. (default: from)",
    "content_tag": "the key in the message represents the content. (default: value)",
    "user_tag": "the value of the role_tag represents the user. (default: human)",
    "assistant_tag": "the value of the role_tag represents the assistant. (default: gpt)",
    "observation_tag": "the value of the role_tag represents the tool results. (default: observation)",
    "function_tag": "the value of the role_tag represents the function call. (default: function_call)",
    "system_tag": "the value of the role_tag represents the system prompt. (default: system, can override system column)"
  }
}
```

## Alpaca Format

### Supervised Fine-Tuning Dataset

* [Example dataset](alpaca_en_demo.json)

In supervised fine-tuning, the `instruction` column will be concatenated with the `input` column and used as the human prompt, then the human prompt would be `instruction\ninput`. The `output` column represents the model response.

The `system` column will be used as the system prompt if specified.

The `history` column is a list consisting of string tuples representing prompt-response pairs in the history messages. Note that the responses in the history **will also be learned by the model** in supervised fine-tuning.

```json
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)",
    "history": [
      ["human instruction in the first round (optional)", "model response in the first round (optional)"],
      ["human instruction in the second round (optional)", "model response in the second round (optional)"]
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
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

### Pre-training Dataset

- [Example dataset](c4_demo.json)

In pre-training, only the `text` column will be used for model learning.

```json
[
  {"text": "document"},
  {"text": "document"}
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "columns": {
    "prompt": "text"
  }
}
```

### Preference Dataset

Preference datasets are used for reward modeling, DPO training and ORPO training.

It requires a better response in `chosen` column and a worse response in `rejected` column.

```json
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "chosen": "chosen answer (required)",
    "rejected": "rejected answer (required)"
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
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

### KTO Dataset

- [Example dataset](kto_en_demo.json)

KTO datasets require a extra `kto_tag` column containing the boolean human feedback.

```json
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "output": "model response (required)",
    "kto_tag": "human feedback [true/false] (required)"
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "kto_tag": "kto_tag"
  }
}
```

### Multimodal Dataset

- [Example dataset](mllm_demo.json)

Multimodal datasets require a `images` column containing the paths to the input images. Currently we only support one image.

```json
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "output": "model response (required)",
    "images": [
      "image path (required)"
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "images": "images"
  }
}
```

## Sharegpt Format

### Supervised Fine-Tuning Dataset

- [Example dataset](glaive_toolcall_en_demo.json)

Compared to the alpaca format, the sharegpt format allows the datasets have **more roles**, such as human, gpt, observation and function. They are presented in a list of objects in the `conversations` column.

Note that the human and observation should appear in odd positions, while gpt and function should appear in even positions.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "human instruction"
      },
      {
        "from": "function_call",
        "value": "tool arguments"
      },
      {
        "from": "observation",
        "value": "tool result"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "tools": "tool description (optional)"
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system",
    "tools": "tools"
  }
}
```

### Preference Dataset

- [Example dataset](dpo_en_demo.json)

Preference datasets in sharegpt format also require a better message in `chosen` column and a worse message in `rejected` column.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "human instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      },
      {
        "from": "human",
        "value": "human instruction"
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "chosen answer (required)"
    },
    "rejected": {
      "from": "gpt",
      "value": "rejected answer (required)"
    }
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
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

### OpenAI Format

The openai format is simply a special case of the sharegpt format, where the first message may be a system prompt.

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "system prompt (optional)"
      },
      {
        "role": "user",
        "content": "human instruction"
      },
      {
        "role": "assistant",
        "content": "model response"
      }
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
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

The KTO datasets and multimodal datasets in sharegpt format are similar to the alpaca format.

Pre-training datasets are **incompatible** with the sharegpt format.
