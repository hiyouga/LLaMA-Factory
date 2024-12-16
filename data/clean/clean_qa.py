# -*- coding: utf-8 -*-
# -------------------------------
# @项目：LLaMA-Factory
# @文件：clean_qa.py
# @时间：2024/12/16 14:54
# @作者：xming
# -------------------------------
import json
import re


def clean_data(input_data):
  # 数据清洗函数
  def clean_text(text):
    if not isinstance(text, str):
      return text
    # 去除多余的换行符
    text = text.strip()
    # 去除重复文本
    text = re.sub(r'^.*?的\s*', '', text)
    return text

  # 保存清洗后的数据
  cleaned_data = []

  # 对原始数据进行处理
  for item in input_data:
    # 深拷贝数据并清洗
    cleaned_item = {
      "INDIVIDUAL_DATA_PROPERTY_NAME": clean_text(
        item.get("INDIVIDUAL_DATA_PROPERTY_NAME", "")),
      "DATA_PROPERTY_VALUE": clean_text(item.get("DATA_PROPERTY_VALUE", ""))
    }
    cleaned_data.append(cleaned_item)

  return cleaned_data

def convert_to_instruction_format(input_data):
  """
  将原始船只数据转换为指令跟随格式

  参数:
  input_data (list): 原始船只数据列表

  返回:
  list: 转换后的指令跟随格式数据列表
  """
  instruction_dataset = []

  for item in input_data:
    input =         item.get("INDIVIDUAL_DATA_PROPERTY_NAME", "")
    value =        item.get("DATA_PROPERTY_VALUE", "")
    instruction_item = {
      "instruction": "根据给定的属性，提供对应的数值",
      "input": input,
      "output": value if value != '/' else '暂无数据'
    }
    instruction_dataset.append(instruction_item)

  return instruction_dataset

# 原始输入数据
# 通过加载本地文件获取输入
with open('实体属性集.json', 'r', encoding='utf-8') as file:
  input_data = json.load(file)

# 转换数据
result = convert_to_instruction_format(input_data)

# 打印结果
# print(json.dumps(result, ensure_ascii=False, indent=2))

# 存储处理后的json到本地文件
with open('../entity.json', 'w', encoding='utf-8') as file:
   json.dump(result, file, ensure_ascii=False, indent=2)
