# -*- coding: utf-8 -*-
# -------------------------------
# @项目：LLaMA-Factory
# @文件：划分疾病与药物数据集.py
# @时间：2025/1/1 15:40
# @作者：xming
# -------------------------------
import codecs
import json
import os
import re

file_path = './疾病与药物.json'

# 使用codecs打开文件，确保正确处理编码
with codecs.open(file_path, 'r', encoding='utf-8') as file:
    # 直接加载JSON数据
    data = json.load(file)

other_data = []
category_mapping = {
    "疾病": {
        "症状": [],
        "治疗方法": []
    },
    "药物": {
        "适应症": [],
        "用法用量": []
    }
}
for item in data:
    dp_name_parts = item["dp_name"].split(": ")
    main_type = dp_name_parts[0].strip()
    sub_type = dp_name_parts[1].split(" 的")[1].strip()

    # 数据清洗
    value_ = item["dp_value"]
    value_ = re.sub('<.*?>', '', value_)
    value_ = re.sub('&lt;.*?&gt;', '', value_)
    value_ = value_.replace('\t', '')
    value_ = value_.replace('\n', '')
    value_ = value_.replace(' ', '')
    if main_type == "疾病":
        if sub_type == "症状":
            category_mapping["疾病"]["症状"].append({
                "instruction": item["dp_name"],
                "input":"",
                "output": value_
            })
        elif sub_type == "治疗方法":
            category_mapping["疾病"]["治疗方法"].append({
                "instruction": item["dp_name"],
                "input": "",
                "output": value_
            })
    elif main_type == "药物":
        if sub_type == "适应症":
            category_mapping["药物"]["适应症"].append({
                "instruction": item["dp_name"],
                "input": "",
                "output": value_
            })
        elif sub_type == "用法与用量":
            category_mapping["药物"]["用法用量"].append({
                "instruction": item["dp_name"],
                "input": "",
                "output": value_
            })
    else:
        other_data.append(item)

if not os.path.exists("../疾病的症状.json"):
    with open("../疾病的症状.json", "w", encoding="utf-8") as f:
        json.dump(category_mapping["疾病"]["症状"], f, ensure_ascii=False, indent=4)

if not os.path.exists("../疾病的治疗方法.json"):
    with open("../疾病的治疗方法.json", "w", encoding="utf-8") as f:
        json.dump(category_mapping["疾病"]["治疗方法"], f, ensure_ascii=False, indent=4)

if not os.path.exists("../药物的适应症.json"):
    with open("../药物的适应症.json", "w", encoding="utf-8") as f:
        json.dump(category_mapping["药物"]["适应症"], f, ensure_ascii=False, indent=4)

if not os.path.exists("../药物的用法用量.json"):
    with open("../药物的用法用量.json", "w", encoding="utf-8") as f:
        json.dump(category_mapping["药物"]["用法用量"], f, ensure_ascii=False, indent=4)

if not os.path.exists("不在上述情况的.json"):
    with open("不在上述情况的.json", "w", encoding="utf-8") as f:
        json.dump(other_data, f, ensure_ascii=False, indent=4)


# result = {}
# for item in data:
#     parts = item["dp_name"].split(": ")
#     category, sub_category = parts[0], parts[1].split(" 的")[0]
#     if category not in result:
#         result[category] = {}
#     if sub_category not in result[category]:
#         result[category][sub_category] = []
#     result[category][sub_category].append({"name": parts[1].split(" 的")[1], "description": item["dp_value"]})
#
# for main_category in result:
#     if not os.path.exists(main_category):
#         os.makedirs(main_category)
#     for sub_category in result[main_category]:
#         with open(f"{main_category}/{sub_category}.json", 'w', encoding='utf-8') as f:
#             json.dump(result[main_category][sub_category], f, ensure_ascii=False, indent=4)