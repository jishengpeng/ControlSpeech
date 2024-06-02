import pandas as pd
import random
import json

data = pd.read_json('/home/PZDS/projects/EI_VC/data/processed/ei_data/metadata.json', encoding='utf-8')

# 遍历所有的item并找到匹配的项
ref_items = {}
for idx, item in data.iterrows():
    sub_dataset = item['sub_dataset']
    spk_name = item['spk_name']
    emotion = item['emotion']
    matching_items = data[(data['sub_dataset'] == sub_dataset) & (data['spk_name'] != spk_name) & (data['emotion'] != emotion)]
    if not matching_items.empty:
        matching_item = matching_items.sample(n=1).iloc[0]
        ref_item = matching_item['item_name']
        ref_items[item['item_name']] = ref_item

print(len(ref_items))
# 将索引关系存入另一个JSON文件中
with open('/home/PZDS/projects/EI_VC/data/processed/ei_data/drive_item.json', 'w',encoding='utf-8') as f:
    json.dump(ref_items, f, indent=4,ensure_ascii=False)


#df.to_csv('/home/yangqian/AdaptSpeech/data/processed/multi_scene/metadata.csv', encoding='utf-8')
