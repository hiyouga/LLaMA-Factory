import json
f = open("/mnt/data/shesj/Data/RL4CoTData/sft_data/sixlan_sft_data.json",'r')
data = json.load(f)

for i in data:
    i['instruction'] = i['instruction'].split("### Instruction:\n")[1].split("\n\n### Response:")[0]

print(data[-1])
f = open("/mnt/data/shesj/Data/RL4CoTData/sft_data/sixlan_sft_data_unwrap.json",'w')
json.dump(data,f,indent=2,ensure_ascii=False)