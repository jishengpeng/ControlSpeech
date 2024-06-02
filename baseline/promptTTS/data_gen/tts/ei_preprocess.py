import json
import os
import random
import re
import traceback
from collections import Counter
from functools import partial
import sys
import re

sys.path.append('')
import librosa
from tqdm import tqdm
from utils.os_utils import link_file, move_file, remove_file


class BasePreprocessor:
    def __init__(self):
        self.dataset_name = 'ei_data_SN'
        self.raw_data_dir = f'/home/PZDS/projects/EI_VC/data/raw/ei_data'
        self.processed_dir = f'data/processed/{self.dataset_name}'
        self.spk_map_fn = f"{self.processed_dir}/spk_map.json"
        self.emotion_map_fp = f"{self.processed_dir}/emotion_map.json"
        self.num_spk = 4000 # 35 for stutter_set and 109 for vctk
        self.seed = 1234
        
    def load_unit(self,unit_fp):
        unit_items = {}
        with open(unit_fp,'r') as fp:
            lines = fp.readlines()
            for l in lines:
                l = l.rstrip('\n')
                last_index = l.rfind('|')  # 找到最后一个 '|' 的索引
                assert last_index != -1,'error!'
                first_part = l[:last_index]  # 划分成两半
                second_part = l[last_index + 1:]
                frames = second_part.split(' ')
                unit_ids = []
                for id in frames:
                    num  = int(id)
                    unit_ids.append(num)
                unit_items[first_part] = unit_ids
        return unit_items

    def meta_data(self):
        # Load dataset info
        unit_fp = '/home/PZDS/projects/EI_VC/data/raw/ei_data_SN/unit.tsv'
        unit_items = self.load_unit(unit_fp)
        root_dir = self.raw_data_dir
        pattern = re.compile(r'^\d+_')
        # {"surprised": 0, "contempt": 1, "sad": 2, "neutral": 3, "disgusted": 4, "happy": 5, "angry": 6, "fear": 7}
        CREMA_D_emotionmap = {'ANG':'angry','DIS':'disgusted','FEA':'fear','HAP':'happy','NEU':'neutral','SAD':'sad'}
        esd_emotionmap = {'Angry':'angry','Happy':'happy','Neutral':'neutral','Sad':'sad','Surprise':'surprised'}
        Mess_emotionmap = {'A':'angry','C':'neutral','H':'happy','S':'sad'}
        RAVDESS_emotionmap = {'01':'neutral','02':'neutral','03':'happy','04':'sad','05':'angry','06':'fear','07':'disgusted','08':'surprised'}
        SAVEE_emotionmap = {'a':'angry','d':'disgusted','sa':'sad','h':'happy','su':'surprised','f':'fear','n':'neutral'}
        TESS_emotionmap = {'angry':'angry','disgust':'disgusted','fear':'fear','happy':'happy','neutral':'neutral','ps':'surprised','sad':'sad'}
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.wav'):
                    item_name = file.split('.')[0]
                    wav_fp = os.path.join(root,file)
                    unit_ids = unit_items.get(item_name,None)
                    if unit_ids==None:
                        continue
                    if item_name.startswith('CREMA-D'):
                        splits = item_name.split('|')[-1].split('_')
                        spk_name = 'CREMA-D'+'-'+splits[0]
                        emotion = CREMA_D_emotionmap[splits[2]]
                        sub_dataset = 'CREMA-D'
                    elif item_name.startswith('Emotional Speech Dataset (ESD)'):
                        splits = item_name.split('|')
                        spk_name = 'ESD'+'-'+splits[-3]
                        emotion = esd_emotionmap[splits[-2]]
                        sub_dataset = 'ESD'
                    elif item_name.startswith('MEAD'):
                        splits= item_name.split('|')
                        spk_name = 'MEAD'+'-'+splits[1]
                        emotion = splits[3]
                        sub_dataset = 'MEAD'
                    elif item_name.startswith('MESS compressed'):
                        splits = item_name.split('|')[-1]
                        spk_name = 'MESS'+'-'+splits[1:3]
                        emotion = Mess_emotionmap[splits[0]]
                        sub_dataset = 'MESS'
                    elif item_name.startswith('RAVDESS'):
                        splits = item_name.split('|')[-1].split('-')
                        spk_name = 'RAVDESS'+'-'+splits[-1]
                        emotion = RAVDESS_emotionmap[splits[2]]
                        sub_dataset = 'RAVDESS'
                    elif item_name.startswith('SAVEE'):
                        splits = item_name.split('|')[-1].split('_')
                        spk_name = 'SAVEE'+'-'+splits[0]
                        emotion = SAVEE_emotionmap[re.findall(r'[a-zA-Z]+', splits[1])[0]]
                        sub_dataset = 'SAVEE'
                    elif item_name.startswith('TESS'):
                        splits = item_name.split('|')[-1].split('_')
                        spk_name = 'TESS'+'-'+splits[0]
                        emotion = TESS_emotionmap[splits[-1]]
                        sub_dataset = 'TESS'
                    elif bool(pattern.match(item_name)):
                        splits = item_name.split('_')
                        spk_name = 'libritts' + '-'+splits[0]
                        emotion = 'neutral'
                        sub_dataset = 'libritts'
                    else:
                        spk_name = None
                        emotion=None
                        unit_ids = None
                        assert 1!=1,'Error metadata info'
                    yield {'item_name':item_name,'wav_fp':wav_fp,'spk_name':spk_name,'emotion':emotion,'unit_ids':unit_ids,'sub_dataset':sub_dataset}


    def process(self):
        processed_dir = self.processed_dir
        remove_file(processed_dir)
        os.makedirs(processed_dir, exist_ok=True)

        meta_data = list(tqdm(self.meta_data(), desc='Load meta data'))
        item_names = [d['item_name'] for d in meta_data]
        assert len(item_names) == len(set(item_names)), 'Key `item_name` should be Unique.'

        # preprocess data
        spk_names = set()
        emotions = set()
        items = []
        
        for item in meta_data:
            spk_names.add(item['spk_name'])
            emotions.add(item['emotion'])
            items.append(item)

        spk_map = self.build_spk_map(spk_names)
        emotion_map = self.build_emotion_map(emotions)
        with open(f"{processed_dir}/{self.meta_csv_filename}.json", 'w') as f:
            f.write(re.sub(r'\n\s+([\d+\]])', r'\1', json.dumps(items, ensure_ascii=False, sort_keys=False, indent=1)))


    def build_spk_map(self, spk_names):
        spk_map = {x: i for i, x in enumerate(sorted(list(spk_names)))}
        assert len(spk_map) == 0 or len(spk_map) <= self.num_spk, len(spk_map)
        print(f"| Number of spks: {len(spk_map)}, spk_map: {spk_map}")
        json.dump(spk_map, open(self.spk_map_fn, 'w'), ensure_ascii=False)
        return spk_map
    
    def build_emotion_map(self, emotions):
        emotion_map = {x: i for i, x in enumerate(sorted(list(emotions)))}
        print(f"| Number of emotions: {len(emotion_map)}, emotion_map: {emotion_map}")
        json.dump(emotion_map, open(self.emotion_map_fp, 'w'), ensure_ascii=False)
        return emotion_map

    @property
    def meta_csv_filename(self):
        return 'metadata'


if __name__ == '__main__':
    BasePreprocessor().process()