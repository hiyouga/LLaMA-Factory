import json
import os.path

import fire
from datasets import Dataset, concatenate_datasets, load_dataset, Value, Image, Features, Sequence

"""usage
python3 scripts/make_mllm_instruct.py \
--json_path data/llava_instruct_example.json \
--image_path data/images \
--output_path data/mllm_example_dataset
"""


def make_one_json(json_path, image_path) -> Dataset:
    with open(json_path) as f:
        raw_data_ls = json.loads(f.read())
    data_ls = []
    for i, data in enumerate(raw_data_ls):
        for j, message in enumerate(data['messages']):
            text = message['content']
            message['content'] = [{'index': None, 'text': text, 'type': 'text'}]
            if j == 0:
                message['content'].append({'index': 0, 'text': None, 'type': 'image'})
        image = data['image']
        if image_path:
            image = os.path.join(image_path, data['image'])
        data['images'] = [image]
        del data['image']
        data_ls.append(data)

    def gen():
        for data in data_ls:
            yield data

    features = Features({'messages': [{'content': [
        {'index': Value(dtype='int64', id=None), 'text': Value(dtype='string', id=None),
         'type': Value(dtype='string', id=None)}], 'role': Value(dtype='string', id=None)}],
        'images': Sequence(feature=Image(decode=True, id=None), length=-1, id=None)})
    dataset = Dataset.from_generator(gen, features=features)
    return dataset


yaml_content = """---
dataset_info:
  features:
  - name: messages
    list:
    - name: content
      list:
      - name: index
        dtype: int64
      - name: text
        dtype: string
      - name: type
        dtype: string
    - name: role
      dtype: string
  - name: images
    sequence: image
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---"""


def main(
    json_path: str,
    image_path: str,
    output_path: str,
):
    json_path_list = json_path.split()
    dataset_list = []
    for json_path in json_path_list:
        dataset = make_one_json(json_path, image_path)
        dataset_list.append(dataset)
    dataset = concatenate_datasets(dataset_list)
    print(dataset[0])
    data_path = os.path.join(output_path, "data")
    os.makedirs(os.path.join(data_path), exist_ok=True)
    parquet_path = os.path.join(data_path, "train-0.parquet")
    dataset.to_parquet(parquet_path)
    parquet_path = os.path.join(data_path, "test-0.parquet")
    dataset.to_parquet(parquet_path)
    readme_path = os.path.join(output_path, "README.md")
    with open(readme_path, 'w') as f:
        f.write(yaml_content)


if __name__ == '__main__':
    fire.Fire(main)
