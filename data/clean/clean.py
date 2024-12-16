# -*- coding: utf-8 -*-
# -------------------------------
# @项目：LLaMA-Factory
# @文件：clean.py
# @时间：2024/12/16 13:59
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


def transform_to_alpaca(cleaned_data):
  # 转换为Alpaca格式
  alpaca_data = []

  # 聚合数据生成instruction和output
  name = next((item['DATA_PROPERTY_VALUE'] for item in cleaned_data if
               '名称' in item['INDIVIDUAL_DATA_PROPERTY_NAME']), '')
  model = next((item['DATA_PROPERTY_VALUE'] for item in cleaned_data if
                '型号' in item['INDIVIDUAL_DATA_PROPERTY_NAME']), '')
  country = next((item['DATA_PROPERTY_VALUE'] for item in cleaned_data if
                  '国家' in item['INDIVIDUAL_DATA_PROPERTY_NAME']), '')
  links = next((item['DATA_PROPERTY_VALUE'] for item in cleaned_data if
                '参考链接' in item['INDIVIDUAL_DATA_PROPERTY_NAME']), '')
  image = next((item['DATA_PROPERTY_VALUE'] for item in cleaned_data if
                '图片' in item['INDIVIDUAL_DATA_PROPERTY_NAME']), '')

  alpaca_item = {
    "instruction": f"提取{name}的详细信息",
    "input": "",
    "output": f"""
名称：{name}
型号：{model}
国家：{country}
参考链接：{links}
图片路径：{image}
""".strip()
  }

  alpaca_data.append(alpaca_item)

  return alpaca_data


def process_data(input_data):
  # 数据清洗
  cleaned_data = clean_data(input_data)

  # 转换为Alpaca格式
  alpaca_data = transform_to_alpaca(cleaned_data)

  return alpaca_data


# 测试数据
test_data = [
  {
    "INDIVIDUAL_DATA_PROPERTY_NAME": "\nBRP康拉多·叶(PS-39)\nBRP Conrado Yap (PS-39) 的 名称",
    "DATA_PROPERTY_VALUE": "\nBRP康拉多·叶(PS-39)\nBRP Conrado Yap (PS-39)"
  },
  {
    "INDIVIDUAL_DATA_PROPERTY_NAME": "\nBRP康拉多·叶(PS-39)\nBRP Conrado Yap (PS-39) 的 型号",
    "DATA_PROPERTY_VALUE": "BRP康拉多·叶(PS-39)号护卫舰"
  },
  {
    "INDIVIDUAL_DATA_PROPERTY_NAME": "\nBRP康拉多·叶(PS-39)\nBRP Conrado Yap (PS-39) 的 国家/地区",
    "DATA_PROPERTY_VALUE": "Philippine 菲律宾"
  },
  {
    "INDIVIDUAL_DATA_PROPERTY_NAME": "\nBRP康拉多·叶(PS-39)\nBRP Conrado Yap (PS-39) 的 参考链接",
    "DATA_PROPERTY_VALUE": "https://mb.com.ph/2023/7/1/centino-inspects-naval-warships-in-subic；https://en.wikipedia.org/wiki/BRP_Conrado_Yap_(PS-39)"
  },
  {
    "INDIVIDUAL_DATA_PROPERTY_NAME": "\nBRP康拉多·叶(PS-39)\nBRP Conrado Yap (PS-39) 的 图片",
    "DATA_PROPERTY_VALUE": "IMAGE/2024/2024-01-06/BRP_Conrado_Yap_(PS-39)/BRP_Conrado_Yap_(PS-39).jpg"
  }
]

# 执行处理
result = process_data(test_data)

# 输出结果
print(json.dumps(result, ensure_ascii=False, indent=2))