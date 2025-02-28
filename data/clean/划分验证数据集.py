import json
import codecs
from sklearn.model_selection import train_test_split


def load_and_process_json(file_path):
  # 使用codecs打开文件，确保正确处理编码
  with codecs.open(file_path, 'r', encoding='utf-8') as file:
    # 直接加载JSON数据
    data = json.load(file)

  # 创建Alpaca格式的数据结构
  alpaca_data = []

  # 处理每个数据条目
  for item in data:
    instruction = item.get('instruction', '')
    input_text = item.get('input', '')
    original_output = item.get('output', '')

    # 创建Alpaca格式条目
    alpaca_entry = {
      "instruction": instruction,
      "input": input_text,
      "output": original_output
    }
    alpaca_data.append(alpaca_entry)

  # 将数据集分割为训练集和测试集（80%训练，20%测试）
  train_data, test_data = train_test_split(alpaca_data, test_size=0.2,
                                           random_state=42)

  # 保存训练集
  with open("../alpaca_train_dataset.json", "w", encoding='utf-8') as file:
    json.dump(train_data, file, indent=4, ensure_ascii=False)

  # 保存测试集
  with open("../alpaca_test_dataset.json", "w", encoding='utf-8') as file:
    json.dump(test_data, file, indent=4, ensure_ascii=False)

  print("Alpaca格式训练数据集已保存为 'alpaca_train_dataset.json'")
  print("Alpaca格式测试数据集已保存为 'alpaca_test_dataset.json'")


# 文件路径
file_path = '../alpaca_dataset.json'

# 加载并处理数据
load_and_process_json(file_path)