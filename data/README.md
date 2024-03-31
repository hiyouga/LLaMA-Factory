If you are using a custom dataset, please provide your dataset definition in the following format in `dataset_info.json`.

```json
"dataset_name": {
  "hf_hub_url": "the name of the dataset repository on the Hugging Face hub. (if specified, ignore script_url and file_name)",
  "ms_hub_url": "the name of the dataset repository on the ModelScope hub. (if specified, ignore script_url and file_name)",
  "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore file_name)",
  "file_name": "the name of the dataset file in this directory. (required if above are not specified)",
  "file_sha1": "the SHA-1 hash value of the dataset file. (optional, does not affect training)",
  "subset": "the name of the subset. (optional, default: None)",
  "folder": "the name of the folder of the dataset repository on the Hugging Face hub. (optional, default: None)",
  "ranking": "whether the dataset is a preference dataset or not. (default: false)",
  "formatting": "the format of the dataset. (optional, default: alpaca, can be chosen from {alpaca, sharegpt})",
  "columns (optional)": {
    "prompt": "the column name in the dataset containing the prompts. (default: instruction)",
    "query": "the column name in the dataset containing the queries. (default: input)",
    "response": "the column name in the dataset containing the responses. (default: output)",
    "history": "the column name in the dataset containing the histories. (default: None)",
    "messages": "the column name in the dataset containing the messages. (default: conversations)",
    "system": "the column name in the dataset containing the system prompts. (default: None)",
    "tools": "the column name in the dataset containing the tool description. (default: None)"
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

Given above, you can use the custom dataset via specifying `--dataset dataset_name`.

----

Currently we support dataset in **alpaca** or **sharegpt** format, the dataset in alpaca format should follow the below format:

```json
[
  {
    "instruction": "user instruction (required)",
    "input": "user input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)",
    "history": [
      ["user instruction in the first round (optional)", "model response in the first round (optional)"],
      ["user instruction in the second round (optional)", "model response in the second round (optional)"]
    ]
  }
]
```

Regarding the above dataset, the `columns` in `dataset_info.json` should be:

```json
"dataset_name": {
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

The `query` column will be concatenated with the `prompt` column and used as the user prompt, then the user prompt would be `prompt\nquery`. The `response` column represents the model response.

The `system` column will be used as the system prompt. The `history` column is a list consisting string tuples representing prompt-response pairs in the history. Note that the responses in the history **will also be used for training**.

For the pre-training datasets, only the `prompt` column will be used for training.

For the preference datasets, the `response` column should be a string list whose length is 2, with the preferred answers appearing first, for example:

```json
{
  "instruction": "user instruction",
  "input": "user input",
  "output": [
    "chosen answer",
    "rejected answer"
  ]
}
```

Remember to set `"ranking": true` for the preference datasets.

----

The dataset in sharegpt format should follow the below format:

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
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

Regarding the above dataset, the `columns` in `dataset_info.json` should be:

```json
"dataset_name": {
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

where the `messages` column should be a list following the `u/a/u/a/u/a` order.

Pre-training datasets and preference datasets are incompatible with the sharegpt format yet.
