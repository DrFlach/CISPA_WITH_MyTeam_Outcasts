---
dataset_info:
- config_name: task
  features:
  - name: path
    dtype: image
  - name: conversation
    list:
    - name: instruction
      dtype: string
    - name: output
      dtype: string
  - name: user_id
    dtype: string
  splits:
  - name: train
    num_bytes: 512215573.0
    num_examples: 1000
  download_size: 511664331
  dataset_size: 512215573.0
- config_name: validation_pii
  features:
  - name: path
    dtype: image
  - name: conversation
    list:
    - name: instruction
      dtype: string
    - name: output
      dtype: string
  - name: user_id
    dtype: string
  splits:
  - name: train
    num_bytes: 145421266.0
    num_examples: 280
  download_size: 145271094
  dataset_size: 145421266.0
- config_name: validation_pii_txt_only
  features:
  - name: path
    dtype: image
  - name: conversation
    list:
    - name: instruction
      dtype: string
    - name: output
      dtype: string
  - name: user_id
    dtype: string
  splits:
  - name: train
    num_bytes: 512232765.0
    num_examples: 1000
  download_size: 511696931
  dataset_size: 512232765.0
configs:
- config_name: task
  data_files:
  - split: train
    path: task/train-*
- config_name: validation_pii
  data_files:
  - split: train
    path: validation_pii/train-*
- config_name: validation_pii_txt_only
  data_files:
  - split: train
    path: validation_pii_txt_only/train-*
---
