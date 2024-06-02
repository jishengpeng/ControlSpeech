import json
import os
import random
import traceback
from functools import partial
import numpy as np
from tqdm import tqdm

import utils.commons.single_thread_env  # NOQA
from utils.audio import librosa_wav2spec
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import remove_file, copy_file

np.seterr(divide='ignore', invalid='ignore')


class BinarizationError(Exception):
    pass


class BaseBinarizer:

    drive_ref_map = json.load(open(f"{hparams['processed_data_dir']}/drive_ref_item.json"))
    items = {}
    item_names = []

    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dir = processed_data_dir
        self.binarization_args = hparams['binarization_args']
        self.items = {}
        self.item_names = []
        

    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            BaseBinarizer.items[item_name] = r
            self.item_names.append(item_name)
            BaseBinarizer.items[item_name] = r

        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        range_ = self._convert_range(self.binarization_args['train_range'])
        return self.item_names[range_[0]:range_[1]]
        #return self.item_names[0:200]

    @property
    def valid_item_names(self):
        range_ = self._convert_range(self.binarization_args['valid_range'])
        return self.item_names[range_[0]:range_[1]]

    @property
    def test_item_names(self):
        range_ = self._convert_range(self.binarization_args['test_range'])
        return self.item_names[range_[0]:range_[1]]

    def _convert_range(self, range_):
        if range_[1] == -1:
            range_[1] = len(self.item_names)
        return range_

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    def process(self):
        self.load_meta_data()
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        for fn in ['emotion_map.json','spk_map.json']:
            remove_file(f"{hparams['binary_data_dir']}/{fn}")
            copy_file(f"{hparams['processed_data_dir']}/{fn}", f"{hparams['binary_data_dir']}/{fn}")
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        meta_data = list(self.meta_data(prefix))    
        process_item = partial(self.process_item, binarization_args=self.binarization_args)
        mel_lengths = []
        total_sec = 0
        items = []
        args = [{'item': item} for item in meta_data]

        # Chunked multiprocessing
        num_samples = len(args)
        num_samples_per_chunk = 5000
        if num_samples <= num_samples_per_chunk:
            num_chunk = 1
            num_samples_per_chunk = num_samples
        else:
            num_chunk = num_samples // num_samples_per_chunk
            num_sample_last_chunk = num_samples % num_samples_per_chunk
            if num_sample_last_chunk != 0:
                num_chunk += 1
        for i_chunk in range(num_chunk):
            items_i = []
            if i_chunk == num_chunk - 1:
                args_i = args[i_chunk*num_samples_per_chunk:]
            else:
                args_i = args[i_chunk*num_samples_per_chunk:(i_chunk+1)*num_samples_per_chunk]

            for item_id, item in multiprocess_run_tqdm(process_item, args_i, desc=f'Processing data of chunk {i_chunk+1}/{num_chunk}', num_workers=10):
                if item is not None:
                    items_i.append(item)

            for item in items_i:
                if not self.binarization_args['with_wav'] and 'wav' in item:
                    del item['wav']
                builder.add_item(item)
                mel_lengths.append(item['len'])
                assert item['len'] > 0, (item['item_name'])
                total_sec += item['sec']

        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, binarization_args):
        item_name = item['item_name']
        wav_fp = item['wav_fp']
        wav, mel = cls.process_audio(wav_fp, item, binarization_args)
        if len(mel) < hparams['min_mel_length']:
            return None
        return item

    @classmethod
    def process_audio(cls, wav_fn, res, binarization_args):
        item_name = res['item_name']
        drive_ref_item_name = cls.drive_ref_map[item_name]
        drive_ref_wavfp = cls.items[drive_ref_item_name]['wav_fp']  
        drive_ref_units = cls.items[drive_ref_item_name]['unit_ids']
        T1 = len(res['unit_ids'])
        T2 = len(drive_ref_units)
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'])
        mel = wav2spec_dict['mel'][:T1]
        drive_ref_wav2spec_dict = librosa_wav2spec(
            drive_ref_wavfp,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'])
        drive_ref_mel = drive_ref_wav2spec_dict['mel'][:T2]
            
        wav = wav2spec_dict['wav'].astype(np.float16)
        drive_wav = drive_ref_wav2spec_dict['wav'].astype(np.float16)
        res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0], 
                    'drive_ref_mel':drive_ref_mel})
        res.update({'drive_ref_units':drive_ref_units})
        return wav, mel
        

    @staticmethod
    def get_spk_embed(wav, ctx):
        return ctx['voice_encoder'].embed_utterance(wav.astype(float))

    @property
    def num_workers(self):
        return int(os.getenv('N_PROC', hparams.get('N_PROC', os.cpu_count())))
