If you are using a custom dataset, please provide your dataset definition in the following format in `dataset_info.json`.

```json
"dataset_name": {
  "hf_hub_url": "the name of the dataset repository on the Hugging Face hub. (if specified, ignore below 3 arguments)",
  "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore below 2 arguments)",
  "file_name": "the name of the dataset file in the this directory. (required if above are not specified)",
  "file_sha1": "the SHA-1 hash value of the dataset file. (optional, does not affect training)",
  "subset": "the name of the subset. (optional, default: None)",
  "ranking": "whether the dataset is a preference dataset or not. (default: false)",
  "formatting": "the format of the dataset. (optional, default: alpaca, can be chosen from {alpaca, sharegpt})",
  "columns": {
    "prompt": "the column name in the dataset containing the prompts. (default: instruction, for alpaca)",
    "query": "the column name in the dataset containing the queries. (default: input, for alpaca)",
    "response": "the column name in the dataset containing the responses. (default: output, for alpaca)",
    "history": "the column name in the dataset containing the histories. (default: None, for alpaca)",
    "messages": "the column name in the dataset containing the messages. (default: conversations, for sharegpt)",
    "role": "the key in the message represents the identity. (default: from, for sharegpt)",
    "content": "the key in the message represents the content. (default: value, for sharegpt)"
  }
}
```

Given above, you can use the custom dataset via specifying `--dataset dataset_name`.

Currently we support dataset in **alpaca** or **sharegpt** format, the dataset in alpaca format should follow the below format:

```json
[
  {
    "instruction": "user instruction (required)",
    "input": "user input (optional)",
    "output": "model response (required)",
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
    "history": "history"
  }
}
```

where the `prompt` and `response` columns should contain non-empty values, represent instruction and response respectively. The `query` column will be concatenated with the `prompt` column and used as input for the model.

The `history` column is a list consisting string tuples representing query-response pairs in history. Note that the responses **in each round will be used for training**.

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
    ]
  }
]
```

Regarding the above dataset, the `columns` in `dataset_info.json` should be:

```json
"dataset_name": {
  "columns": {
    "messages": "conversations",
    "role": "from",
    "content": "value"
  }
}
```

where the `messages` column should be a list whose length is even, and follow the `u/a/u/a/u/a` order.

Pre-training datasets and preference datasets are incompatible with the sharegpt format yet.
