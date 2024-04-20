import json
f = open("/mnt/data/shesj/Data/RL4CoTData/rm_data/sixlan_align_data_5k_Merged_MisConInstructLowSix_0.7_20en_collect-4ensemble-reset-dev.json",'r')
data = json.load(f)

parsed_data = []

from tqdm import tqdm
for i in tqdm(data):
    instruction = i['accept'].split("### Instruction:\n")[1].split("\n\n### Response:")[0]
    parsed_data.append({"instruction":instruction,"input":"",'output':[i['accept'].split("### Response:")[1],i['reject'].split("### Response:")[1]]})

print(len(parsed_data))
f = open("/mnt/data/shesj/Data/LFData/sixlan_align_data_5k_Merged_MisConInstructLowSix_0.7_20en_collect-4ensemble-reset-dev.json",'w')
json.dump(parsed_data,f,indent=4,ensure_ascii=False)
