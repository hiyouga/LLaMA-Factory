---
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
---