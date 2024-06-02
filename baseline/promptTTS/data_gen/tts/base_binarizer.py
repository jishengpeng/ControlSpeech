import json
import os
import random
import traceback
from functools import partial

import numpy as np
from resemblyzer import VoiceEncoder
from tqdm import tqdm

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
    def __init__(self, processed_data_dir='data/processed/stutter_set', binary_data_dir='data/binary/stutter_set'):
        self.dataset_name = 'stutter_set'
        self.processed_data_dir = processed_data_dir
        self.binary_data_dir = binary_data_dir
        self.items = {}
        self.item_names = []
        self.shuffle = True
        self.with_spk_embed = True
        self.with_wav = False

        # text2mel parameters
        self.text2mel_params = {'fft_size': 1024, 'hop_size': 256, 'win_size': 1024,
                                'audio_num_mel_bins': 80, 'fmin': 55, 'fmax': 7600,
                                'f0_min': 80, 'f0_max': 600, 'pitch_extractor': 'parselmouth',
                                'audio_sample_rate': 22050, 'loud_norm': False,
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
        for fn in ['phone_set.json', 'word_set.json', 'spk_map.json']:
            remove_file(f"{self.binary_data_dir}/{fn}")
            copy_file(f"{self.processed_data_dir}/{fn}", f"{self.binary_data_dir}/{fn}")
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')

    def process_data(self, prefix):
        data_dir = self.binary_data_dir
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
        meta_data = list(self.meta_data(prefix))
        process_item = partial(self.process_item)
        ph_lengths = []
        mel_lengths = []
        total_sec = 0
        items = []
        args = [{'item': item, 'text2mel_params': self.text2mel_params} for item in meta_data]
        for item_id, item in multiprocess_run_tqdm(process_item, args, desc='Processing data'):
            if item is not None:
                items.append(item)
        if self.with_spk_embed:
            args = [{'wav': item['wav']} for item in items]
            for item_id, spk_embed in multiprocess_run_tqdm(
                    self.get_spk_embed, args,
                    init_ctx_func=lambda wid: {'voice_encoder': VoiceEncoder().cuda()}, num_workers=2,
                    desc='Extracting spk embed'):
                items[item_id]['spk_embed'] = spk_embed
                if spk_embed is None:
                    del items[item_id]

        for item in items:
            if not self.with_wav and 'wav' in item:
                del item['wav']
            builder.add_item(item)
            mel_lengths.append(item['len'])
            assert item['len'] > 0, (item['item_name'], item['txt'], item['mel2ph'])
            if 'ph_len' in item:
                ph_lengths.append(item['ph_len'])
            total_sec += item['sec']
        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
        if len(ph_lengths) > 0:
            np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
        print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, text2mel_params):
        item['ph_len'] = len(item['ph_token'])
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, text2mel_params)
        if len(mel) < text2mel_params['min_mel_length']:
            return None
        try:
            # stutter label
            cls.process_stutter_label(wav, mel, item, text2mel_params)
            # alignments
            n_bos_frames, n_eos_frames = 0, 0
            if text2mel_params['with_align']:
                tg_fn = f"data/processed/{text2mel_params['dataset_name']}/mfa_outputs/{item_name}.TextGrid"
                item['tg_fn'] = tg_fn
                cls.process_align(tg_fn, item, text2mel_params)
                if text2mel_params['trim_eos_bos']:
                    n_bos_frames = item['dur'][0]
                    n_eos_frames = item['dur'][-1]
                    T = len(mel)
                    item['mel'] = mel[n_bos_frames:T - n_eos_frames]
                    item['mel2ph'] = item['mel2ph'][n_bos_frames:T - n_eos_frames]
                    item['mel2word'] = item['mel2word'][n_bos_frames:T - n_eos_frames]
                    item['dur'] = item['dur'][1:-1]
                    item['dur_word'] = item['dur_word'][1:-1]
                    item['len'] = item['mel'].shape[0]
                    item['wav'] = wav[n_bos_frames * text2mel_params['hop_size']:len(wav) - n_eos_frames * text2mel_params['hop_size']]
            if text2mel_params['with_f0']:
                cls.process_pitch(item, n_bos_frames, n_eos_frames, text2mel_params)
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
    
    @classmethod
    def process_stutter_label(cls, wav, mel, res, text2mel_params):
        # obtain the stutter-oriented mel mask from stutter_label
        stutter_fn = f"data/processed/stutter_set/stutter_labels/{res['item_name'][:17]}/{res['item_name']}.npy"
        stutter_label = np.load(stutter_fn)
        stutter_mel_mask = np.zeros(mel.shape[0])
        if len(stutter_label) > 0:
            for item in stutter_label:
                stutter_start_time, stutter_end_time = item[0], item[1]
                stutter_start_frame = int(stutter_start_time * text2mel_params['audio_sample_rate'] // text2mel_params['hop_size'])
                stutter_end_frame = int(stutter_end_time * text2mel_params['audio_sample_rate'] // text2mel_params['hop_size'])
                if item[2] != 0:
                    item[2] = 1
                stutter_mel_mask[stutter_start_frame:stutter_end_frame] = item[2]
        res.update({'stutter_mel_mask': stutter_mel_mask})

    @staticmethod
    def process_align(tg_fn, item, text2mel_params):
        ph = item['ph']
        mel = item['mel']
        ph_token = item['ph_token']
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, text2mel_params['hop_size'], text2mel_params['audio_sample_rate'],
                                     text2mel_params['mfa_min_sil_duration'])
        else:
            raise BinarizationError(f"Align not found")
        if np.array(mel2ph).max() - 1 >= len(ph_token):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {np.array(mel2ph).max() - 1}, len(phone_encoded): {len(ph_token)}")
        item['mel2ph'] = mel2ph
        item['dur'] = dur

        ph2word = item['ph2word']
        mel2word = [ph2word[p - 1] for p in item['mel2ph']]
        item['mel2word'] = mel2word  # [T_mel]
        dur_word = mel2token_to_dur(mel2word, len(item['word_token']))
        item['dur_word'] = dur_word.tolist()  # [T_word]

    @staticmethod
    def process_pitch(item, n_bos_frames, n_eos_frames, text2mel_params):
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
