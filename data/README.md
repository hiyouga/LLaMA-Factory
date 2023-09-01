If you are using a custom dataset, please provide your dataset definition in the following format in `dataset_info.json`.

```json
"dataset_name": {
  "hf_hub_url": "the name of the dataset repository on the HuggingFace hub. (if specified, ignore below 3 arguments)",
  "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore below 2 arguments)",
  "file_name": "the name of the dataset file in the this directory. (required if above are not specified)",
  "file_sha1": "the SHA-1 hash value of the dataset file. (optional)",
  "ranking": "whether the examples contains ranked responses or not. (default: false)",
  "columns": {
    "prompt": "the name of the column in the datasets containing the prompts. (default: instruction)",
    "query": "the name of the column in the datasets containing the queries. (default: input)",
    "response": "the name of the column in the datasets containing the responses. (default: output)",
    "history": "the name of the column in the datasets containing the history of chat. (default: None)"
  }
}
```

where the `prompt` and `response` columns should contain non-empty values. The `query` column will be concatenated with the `prompt` column and used as input for the model. The `history` column should contain a list where each element is a string tuple representing a query-response pair.

For datasets used in reward modeling or DPO training, the `response` column should be a string list, with the preferred answers appearing first, for example:

```json
{
  "instruction": "Question",
  "input": "",
  "output": [
    "Chosen answer",
    "Rejected answer"
  ]
}
```
