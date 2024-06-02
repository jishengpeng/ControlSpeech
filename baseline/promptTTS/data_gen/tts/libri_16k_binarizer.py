import json
import os
import random
import traceback
from functools import partial

import numpy as np
from resemblyzer import VoiceEncoder
from tqdm import tqdm
import sys
sys.path.append('')

import utils.commons.single_thread_env  # NOQA
from utils.audio import librosa_wav2spec
from utils.audio.align import get_mel2ph, mel2token_to_dur
from utils.audio.cwt import get_lf0_cwt, get_cont_lf0
from utils.audio.pitch.utils import f0_to_coarse
from utils.audio.pitch_extractors import extract_pitch
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.os_utils import remove_file, copy_file

np.seterr(divide='ignore', invalid='ignore')


class BinarizationError(Exception):
    pass


class BaseBinarizer:
    def __init__(self, processed_data_dir='data/processed/libritts_16k', binary_data_dir='data/binary/libritts_16k'):
        self.dataset_name = 'libritts_16k'
        self.processed_data_dir = processed_data_dir
        self.binary_data_dir = binary_data_dir
        self.items = {}
        self.item_names = []
        self.shuffle = True
        self.with_wav = True

        # text2mel parameters
        self.text2mel_params = {'fft_size': 1024, 'hop_size': 320, 'win_size': 1024,
                                'audio_num_mel_bins': 80, 'fmin': 55, 'fmax': 7600,
                                'f0_min': 80, 'f0_max': 600, 'pitch_extractor': 'parselmouth',
                                'audio_sample_rate': 16000, 'loud_norm': False,
                                'mfa_min_sil_duration': 0.1, 'trim_eos_bos': False,
                                'with_align': True, 'text2mel_params': False,
                                'dataset_name': self.dataset_name,
                                'with_f0': True, 'min_mel_length': 64}

    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.shuffle:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        range_ = self._convert_range([400, -1])
        return self.item_names[range_[0]:range_[1]]

    @property
    def valid_item_names(self):
        range_ = self._convert_range([0, 400])
        return self.item_names[range_[0]:range_[1]]

    @property
    def test_item_names(self):
        range_ = self._convert_range([0, 400])
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
        os.makedirs(self.binary_data_dir, exist_ok=True)
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = self.binary_data_dir
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        meta_data = list(self.meta_data(prefix))
        process_item = partial(self.process_item)
        mel_lengths = []
        total_sec = 0
        items = []
        args = [{'item': item, 'text2mel_params': self.text2mel_params} for item in meta_data]
        for item_id, item in multiprocess_run_tqdm(process_item, args, desc='Processing data'):
            if item is not None:
                items.append(item)

        for item in items:
            if not self.with_wav and 'wav' in item:
                del item['wav']
            builder.add_item(item)
            mel_lengths.append(item['len'])
            assert item['len'] > 0, (item['item_name'])
            total_sec += item['sec']
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, text2mel_params):
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, text2mel_params)
        if len(mel) < text2mel_params['min_mel_length']:
            return None
        try:
            cls.process_pitch(item, text2mel_params)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except Exception as e:
            traceback.print_exc()
            print(f"| Skip item. item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return item

    @classmethod
    def process_audio(cls, wav_fn, res, text2mel_params):
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=text2mel_params['fft_size'],
            hop_size=text2mel_params['hop_size'],
            win_length=text2mel_params['win_size'],
            num_mels=text2mel_params['audio_num_mel_bins'],
            fmin=text2mel_params['fmin'],
            fmax=text2mel_params['fmax'],
            sample_rate=text2mel_params['audio_sample_rate'],
            loud_norm=text2mel_params['loud_norm'])
        mel = wav2spec_dict['mel']
        wav = wav2spec_dict['wav'].astype(np.float16)
        # if binarization_args['with_linear']:
        #     res['linear'] = wav2spec_dict['linear']
        res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / text2mel_params['audio_sample_rate'], 'len': mel.shape[0]})
        return wav, mel
    

    @staticmethod
    def process_pitch(item, text2mel_params):
        wav, mel = item['wav'], item['mel']
        f0 = extract_pitch(text2mel_params['pitch_extractor'], wav,
                         text2mel_params['hop_size'], text2mel_params['audio_sample_rate'],
                         f0_min=text2mel_params['f0_min'], f0_max=text2mel_params['f0_max'])
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        assert len(mel) == len(f0), (len(mel), len(f0))
        pitch_coarse = f0_to_coarse(f0)
        item['f0'] = f0
        item['pitch'] = pitch_coarse
        # if hparams['binarization_args']['with_f0cwt']:
        #     uv, cont_lf0_lpf = get_cont_lf0(f0)
        #     logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
        #     cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
        #     cwt_spec, scales = get_lf0_cwt(cont_lf0_lpf_norm)
        #     item['cwt_spec'] = cwt_spec
        #     item['cwt_mean'] = logf0s_mean_org
        #     item['cwt_std'] = logf0s_std_org

    @staticmethod
    def get_spk_embed(wav, ctx):
        return ctx['voice_encoder'].embed_utterance(wav.astype(float))

    @property
    def num_workers(self):
        return int(os.getenv('N_PROC', hparams.get('N_PROC', os.cpu_count())))


if __name__ == '__main__':
    BaseBinarizer().process()
