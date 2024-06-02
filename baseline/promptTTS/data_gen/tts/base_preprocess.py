import json
import os
import random
import re
import traceback
from collections import Counter
from functools import partial

import librosa
from tqdm import tqdm
from data_gen.tts.txt_processors.en import TxtProcessor
from data_gen.tts.wav_processors.base_processor import get_wav_processor_cls
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import link_file, move_file, remove_file
from utils.text.text_encoder import is_sil_phoneme, build_token_encoder


class BasePreprocessor:
    def __init__(self):
        self.txt_processor = TxtProcessor()
        self.dataset_name = 'stutter_set'
        # self.raw_data_dir = f'data/raw/{self.dataset_name}'
        # self.raw_data_dir = f'/home/jzy/SyntaSpeech/data/raw/LibriTTS'
        self.raw_data_dir = f'data/raw/stutter_set'
        self.processed_dir = f'data/processed/{self.dataset_name}'
        self.spk_map_fn = f"{self.processed_dir}/spk_map.json"
        self.reset_phone_dict = True
        self.reset_word_dict = True
        self.word_dict_size = 12500
        self.num_spk = 1200 # 35 for stutter_set and 109 for vctk
        self.use_mfa = True
        self.seed = 1234
        self.nsample_per_mfa_group = 1000
        self.mfa_group_shuffle = False
        self.wav_processors = []

    def meta_data(self):
        # Load dataset info (stutter_set)
        if self.dataset_name == 'stutter_set':
            # Load spk info
            tmp_spk_dict = {}
            with open(f'{self.raw_data_dir}/video_spk.txt', 'r') as f:
                spk_metadata = f.readlines()
            for line in spk_metadata:
                video_name, spk_name = line.split(' ')[0], line.split(' ')[1]
                tmp_spk_dict[video_name] = spk_name
            # Load dataset items
            with open(f"{self.raw_data_dir}/metadata.csv", 'r') as f:
                metadata_lines = f.readlines()
            for r in metadata_lines:
                item_name = r.split('|')[0].split('/')[-1][:-4]
                wav_fn = r.split('|')[0]
                txt = r.split('|')[1].replace('\n', '')
                video_id = item_name[0:13]
                spk_name = tmp_spk_dict[video_id]
                yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}
        # Load dataset info (vctk)
        elif self.dataset_name == 'vctk':
            from glob import glob
            # Load dataset items
            wav_fns = glob(f'data/raw/VCTK-Corpus/wav48/*/*.wav')
            for wav_fn in wav_fns:
                item_name = os.path.basename(wav_fn)[:-4]
                spk_name = wav_fn.split('/')[-2]
                txt_fn = wav_fn.split("/")
                txt_fn[-1] = f'{item_name}.txt'
                txt_fn[-3] = f'txt'
                txt_fn = "/".join(txt_fn)
                if os.path.exists(txt_fn) and os.path.exists(wav_fn):
                    with open(txt_fn, 'r') as f:
                        txt = f.read()
                    yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}
        elif self.dataset_name == 'libritts':
            from glob import glob
            wav_fns = sorted(glob(f'{self.raw_data_dir}/*/*/*/*.wav'))
            for wav_fn in wav_fns:
                item_name = os.path.basename(wav_fn)[:-4]
                txt_fn = f'{wav_fn[:-4]}.normalized.txt'
                with open(txt_fn, 'r') as f:
                    txt = f.read()
                spk_name = item_name.split("_")[0]
                yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk_name}

    def process(self):
        processed_dir = self.processed_dir
        wav_processed_tmp_dir = f'{processed_dir}/processed_tmp'
        remove_file(wav_processed_tmp_dir)
        os.makedirs(wav_processed_tmp_dir, exist_ok=True)
        wav_processed_dir = f'{processed_dir}/{self.wav_processed_dirname}'
        remove_file(wav_processed_dir)
        os.makedirs(wav_processed_dir, exist_ok=True)

        meta_data = list(tqdm(self.meta_data(), desc='Load meta data'))
        item_names = [d['item_name'] for d in meta_data]
        assert len(item_names) == len(set(item_names)), 'Key `item_name` should be Unique.'

        # preprocess data
        phone_list = []
        word_list = []
        spk_names = set()
        process_item = partial(self.preprocess_first_pass,
                               txt_processor=self.txt_processor,
                               wav_processed_dir=wav_processed_dir,
                               wav_processed_tmp=wav_processed_tmp_dir)
        items = []
        args = [{
            'item_name': item_raw['item_name'],
            'txt_raw': item_raw['txt'],
            'wav_fn': item_raw['wav_fn'],
            'txt_loader': item_raw.get('txt_loader'),
            'others': item_raw.get('others', None)
        } for item_raw in meta_data]
        for item_, (item_id, item) in zip(meta_data, multiprocess_run_tqdm(process_item, args, desc='Preprocess')):
            if item is not None:
                item_.update(item)
                item = item_
                if 'txt_loader' in item:
                    del item['txt_loader']
                item['id'] = item_id
                item['spk_name'] = item.get('spk_name', '<SINGLE_SPK>')
                item['others'] = item.get('others', None)
                phone_list += item['ph'].split(" ")
                word_list += item['word'].split(" ")
                spk_names.add(item['spk_name'])
                items.append(item)

        # add encoded tokens
        ph_encoder, word_encoder = self._phone_encoder(phone_list), self._word_encoder(word_list)
        spk_map = self.build_spk_map(spk_names)
        args = [{
            'ph': item['ph'], 'word': item['word'], 'spk_name': item['spk_name'],
            'word_encoder': word_encoder, 'ph_encoder': ph_encoder, 'spk_map': spk_map
        } for item in items]
        for idx, item_new_kv in multiprocess_run_tqdm(self.preprocess_second_pass, args, desc='Add encoded tokens'):
            items[idx].update(item_new_kv)

        # build mfa data
        if self.use_mfa:
            mfa_dict = set()
            mfa_input_dir = f'{processed_dir}/mfa_inputs'
            remove_file(mfa_input_dir)
            # group MFA inputs for better parallelism
            mfa_groups = [i // self.nsample_per_mfa_group for i in range(len(items))]
            if self.mfa_group_shuffle:
                random.seed(self.seed)
                random.shuffle(mfa_groups)
            args = [{
                'item': item, 'mfa_input_dir': mfa_input_dir,
                'mfa_group': mfa_group, 'wav_processed_tmp': wav_processed_tmp_dir
            } for item, mfa_group in zip(items, mfa_groups)]
            for i, (ph_gb_word_nosil, new_wav_align_fn) in multiprocess_run_tqdm(
                    self.build_mfa_inputs, args, desc='Build MFA data'):
                items[i]['wav_align_fn'] = new_wav_align_fn
                for w in ph_gb_word_nosil.split(" "):
                    mfa_dict.add(f"{w} {w.replace('_', ' ')}")
            mfa_dict = sorted(mfa_dict)
            with open(f'{processed_dir}/mfa_dict.txt', 'w') as f:
                f.writelines([f'{l}\n' for l in mfa_dict])
        with open(f"{processed_dir}/{self.meta_csv_filename}.json", 'w') as f:
            f.write(re.sub(r'\n\s+([\d+\]])', r'\1', json.dumps(items, ensure_ascii=False, sort_keys=False, indent=1)))
        remove_file(wav_processed_tmp_dir)

    @classmethod
    def preprocess_first_pass(cls, item_name, txt_raw, txt_processor,
                              wav_fn, wav_processed_dir, wav_processed_tmp,
                              txt_loader=None, others=None):
        try:
            if txt_loader is not None:
                txt_raw = txt_loader(txt_raw)
            ph, txt, word, ph2word, ph_gb_word = cls.txt_to_ph(txt_processor, txt_raw)
            wav_align_fn = wav_fn
            # wav, _ = librosa.core.load(wav_fn, sr=22050)
            # if wav.shape[0] > 176400 or wav.shape[0] < 44100:
            #     print(f"| The audio length > 8s or < 2s. item_name: {item_name}.")
            #     return None
            # wav for binarization
            ext = os.path.splitext(wav_fn)[1]
            os.makedirs(wav_processed_dir, exist_ok=True)
            new_wav_fn = f"{wav_processed_dir}/{item_name}{ext}"
            move_link_func = move_file if os.path.dirname(wav_fn) == wav_processed_tmp else link_file
            move_link_func(wav_fn, new_wav_fn)

            return {
                'txt': txt, 'txt_raw': txt_raw, 'ph': ph,
                'word': word, 'ph2word': ph2word, 'ph_gb_word': ph_gb_word,
                'wav_fn': new_wav_fn, 
                'wav_align_fn': wav_align_fn,
                'others': others
            }
        except:
            traceback.print_exc()
            print(f"| Error is caught. item_name: {item_name}.")
            return None

    @staticmethod
    def txt_to_ph(txt_processor, txt_raw):
        txt_struct, txt = txt_processor.process(txt_raw)
        ph = [p for w in txt_struct for p in w[1]]
        ph_gb_word = ["_".join(w[1]) for w in txt_struct]
        words = [w[0] for w in txt_struct]
        # word_id=0 is reserved for padding
        ph2word = [w_id + 1 for w_id, w in enumerate(txt_struct) for _ in range(len(w[1]))]
        return " ".join(ph), txt, " ".join(words), ph2word, " ".join(ph_gb_word)
    
    def _phone_encoder(self, ph_set):
        ph_set_fn = f"{self.processed_dir}/phone_set.json"
        if self.reset_phone_dict or not os.path.exists(ph_set_fn):
            ph_set = sorted(set(ph_set))
            json.dump(ph_set, open(ph_set_fn, 'w'), ensure_ascii=False)
            print("| Build phone set: ", ph_set)
        else:
            ph_set = json.load(open(ph_set_fn, 'r'))
            print("| Load phone set: ", ph_set)
        return build_token_encoder(ph_set_fn)

    def _word_encoder(self, word_set):
        word_set_fn = f"{self.processed_dir}/word_set.json"
        if self.reset_word_dict:
            word_set = Counter(word_set)
            total_words = sum(word_set.values())
            word_set = word_set.most_common(self.word_dict_size)
            num_unk_words = total_words - sum([x[1] for x in word_set])
            word_set = ['<BOS>', '<EOS>'] + [x[0] for x in word_set]
            word_set = sorted(set(word_set))
            json.dump(word_set, open(word_set_fn, 'w'), ensure_ascii=False)
            print(f"| Build word set. Size: {len(word_set)}, #total words: {total_words},"
                  f" #unk_words: {num_unk_words}, word_set[:10]:, {word_set[:10]}.")
        else:
            word_set = json.load(open(word_set_fn, 'r'))
            print("| Load word set. Size: ", len(word_set), word_set[:10])
        return build_token_encoder(word_set_fn)

    @classmethod
    def preprocess_second_pass(cls, word, ph, spk_name, word_encoder, ph_encoder, spk_map):
        word_token = word_encoder.encode(word)
        ph_token = ph_encoder.encode(ph)
        spk_id = spk_map[spk_name]
        return {'word_token': word_token, 'ph_token': ph_token, 'spk_id': spk_id}

    def build_spk_map(self, spk_names):
        spk_map = {x: i for i, x in enumerate(sorted(list(spk_names)))}
        assert len(spk_map) == 0 or len(spk_map) <= self.num_spk, len(spk_map)
        print(f"| Number of spks: {len(spk_map)}, spk_map: {spk_map}")
        json.dump(spk_map, open(self.spk_map_fn, 'w'), ensure_ascii=False)
        return spk_map
    
    @classmethod
    def build_mfa_inputs(cls, item, mfa_input_dir, mfa_group, wav_processed_tmp):
        item_name = item['item_name']
        wav_align_fn = item['wav_align_fn']
        ph_gb_word = item['ph_gb_word']
        ext = os.path.splitext(wav_align_fn)[1]
        mfa_input_group_dir = f'{mfa_input_dir}/{mfa_group}'
        os.makedirs(mfa_input_group_dir, exist_ok=True)
        new_wav_align_fn = f"{mfa_input_group_dir}/{item_name}{ext}"
        move_link_func = move_file if os.path.dirname(wav_align_fn) == wav_processed_tmp else link_file
        move_link_func(wav_align_fn, new_wav_align_fn)
        ph_gb_word_nosil = " ".join(["_".join([p for p in w.split("_") if not is_sil_phoneme(p)])
                                     for w in ph_gb_word.split(" ") if not is_sil_phoneme(w)])
        with open(f'{mfa_input_group_dir}/{item_name}.lab', 'w') as f_txt:
            f_txt.write(ph_gb_word_nosil)
        return ph_gb_word_nosil, new_wav_align_fn

    def load_spk_map(self, base_dir):
        spk_map_fn = f"{base_dir}/spk_map.json"
        spk_map = json.load(open(spk_map_fn, 'r'))
        return spk_map

    def load_dict(self, base_dir):
        ph_encoder = build_token_encoder(f'{base_dir}/phone_set.json')
        word_encoder = build_token_encoder(f'{base_dir}/word_set.json')
        return ph_encoder, word_encoder

    @property
    def meta_csv_filename(self):
        return 'metadata'

    @property
    def wav_processed_dirname(self):
        return 'wav_processed'


if __name__ == '__main__':
    BasePreprocessor().process()