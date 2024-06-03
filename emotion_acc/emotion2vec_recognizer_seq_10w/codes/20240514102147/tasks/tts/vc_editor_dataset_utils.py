import torch.optim
import torch.utils.data
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributions
import random
import json
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset
from utils.spec_aug.time_mask  import random_crop_with_prob


class BaseSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == 'train' and hparams['min_frames'] > 0:
                self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]
            print(f'{len(self.sizes)} num samples')

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item['mel']) == self.sizes[index], (len(item['mel']), self.sizes[index])
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0)
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'mels': mels,
            'mel_lengths': mel_lengths,
        }
        return batch

class FastSpeechDataset(BaseSpeechDataset):
    def __getitem__(self, index):
        sample = super(FastSpeechDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample['mel']
        T = mel.shape[0]
        ph_token = sample['txt_token']
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T]
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if hparams['pitch_type'] == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)
        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(FastSpeechDataset, self).collater(samples)
        hparams = self.hparams
        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            pitch = collate_1d_or_2d([s['pitch'] for s in samples])
            uv = collate_1d_or_2d([s['uv'] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        mel2ph = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        batch.update({
            'mel2ph': mel2ph,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        })
        return batch
 
class EIEditorDataset(BaseSpeechDataset):
    def __getitem__(self, index):
        sample = super(EIEditorDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample['mel']
        T = mel.shape[0]
        max_frames = hparams['max_frames']
        drive_ref_mel = torch.Tensor(item['drive_ref_mel'])[:max_frames]
        unit_ids = torch.LongTensor(item['unit_ids'])[:T]
        drive_ref_units = torch.LongTensor(item['drive_ref_units'])[:drive_ref_mel.shape[0]]
        sample['drive_ref_mel'] = drive_ref_mel
        sample['unit_ids'] = unit_ids
        sample['drive_ref_units'] = drive_ref_units
        sample['spk_name'] = item['spk_name']
        sample['emotion'] = item['emotion']
        sample['sub_dataset'] = item['sub_dataset']
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(EIEditorDataset, self).collater(samples)
        hparams = self.hparams
        unit_ids = collate_1d_or_2d([s['unit_ids'] for s in samples], 100)
        drive_ref_units = collate_1d_or_2d([s['drive_ref_units'] for s in samples], 100)
        drive_ref_mels = collate_1d_or_2d([s['drive_ref_mel'] for s in samples], 0.0)
        drive_mel_lengths = torch.LongTensor([s['drive_ref_mel'].shape[0] for s in samples])
        
        spk_names = [s['spk_name'] for s in samples]
        emotions = [s['emotion'] for s in samples]
        sub_datasets = [s['sub_dataset'] for s in samples]
        batch.update({
            'unit_ids': unit_ids,
            'drive_ref_units': drive_ref_units,
            'drive_ref_mels': drive_ref_mels,
            'drive_mel_lengths': drive_mel_lengths,
            'spk_names': spk_names,
            'emotions': emotions,
            'sub_datasets': sub_datasets,
        })
        return batch

class VCCLAPEditorDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix,shuffle,items,data_dir)
        with open(f'./{self.data_dir}/spk_map.json', 'r') as file:
            spk_dict = json.load(file)
        self.spk_dict = spk_dict
        
    def __getitem__(self, index):
        sample = super(VCCLAPEditorDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample['mel']
        T = mel.shape[0]
        max_frames = hparams['max_frames']
        ph_token = item['ph_token']
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T]
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if hparams['pitch_type'] == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)
        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        
        if hparams.get('parallel',False):
            ref_spk_mel = torch.Tensor(item['mel'])[:max_frames]
        else:
            ref_spk_mel = torch.Tensor(item['ref_spk_mel'])[:max_frames]
        if hparams.get('use_random',False):
            if random.random() < 0.5:
                ref_spk_mel = torch.Tensor(item['mel'])[:max_frames]
            else:
                ref_spk_mel = torch.Tensor(item['ref_spk_mel'])[:max_frames]
        
        spk_id = self.spk_dict[item['spk_name']]
        sample['spk_id'] = spk_id
        sample['ref_spk_mel'] = ref_spk_mel
        sample['spk_name'] = item['spk_name']
        sample['emotion'] = item['emotion']
        sample['sub_dataset'] = item['sub_dataset']
        sample['item_name'] = item['item_name']
        sample['style_embed'] = torch.Tensor(item['style_embed'])
        sample['ph_token'] = torch.LongTensor(item['ph_token'])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(VCCLAPEditorDataset, self).collater(samples)
        hparams = self.hparams
        ph_tokens = collate_1d_or_2d([s['ph_token'] for s in samples], 100)
        ref_spk_mels = collate_1d_or_2d([s['ref_spk_mel'] for s in samples], 0.0)
        spk_mel_lengths = torch.LongTensor([s['ref_spk_mel'].shape[0] for s in samples])
        mel2ph = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        style_embed = collate_1d_or_2d([s['style_embed'] for s in samples], 0.0)
        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            uv = collate_1d_or_2d([s['uv'] for s in samples])
            pitch = collate_1d_or_2d([s['pitch'] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        
        item_names = [s['item_name'] for s in samples]
        spk_names = [s['spk_name'] for s in samples]
        spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
        emotions = [s['emotion'] for s in samples]
        sub_datasets = [s['sub_dataset'] for s in samples]
        batch.update({
            'item_names': item_names,
            'ph_tokens': ph_tokens,
            'ref_spk_mels': ref_spk_mels,
            'spk_mel_lengths': spk_mel_lengths,
            'spk_names': spk_names,
            'emotions': emotions,
            'sub_datasets': sub_datasets,
            'mel2ph':mel2ph,
            'f0': f0,
            'uv': uv,
            'pitch': pitch,
            'style_embed':style_embed,
            'spk_ids':spk_ids,
        })
        return batch

class TCTTSEditorDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix,shuffle,items,data_dir)
        with open(f'./{self.data_dir}/spk_map.json', 'r') as file:
            spk_dict = json.load(file)
        self.spk_dict = spk_dict
        
    def __getitem__(self, index):
        sample = super(TCTTSEditorDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample['mel']
        T = mel.shape[0]
        max_frames = hparams['max_frames']
        ph_token = item['ph_token']
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T]
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if hparams['pitch_type'] == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)
        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        

        if hparams.get('parallel',False):
            ### 可以考虑使用target的mel，但是提供word-level的pooling，，以及slice部分word的embedding
            if hparams.get('use_crop',False):
                ref_spk_mel = random_crop_with_prob(torch.Tensor(item['mel'])[:max_frames],0.8)
            else:
                ref_spk_mel = torch.Tensor(item['mel'])[:max_frames]
        else:
            ref_spk_mel = torch.Tensor(item['ref_spk_mel'])[:max_frames]
        if hparams.get('use_random',False):
            if random.random() < 0.5:
                ref_spk_mel = torch.Tensor(item['mel'])[:max_frames]
            else:
                ref_spk_mel = torch.Tensor(item['ref_spk_mel'])[:max_frames]
        
        spk_id = self.spk_dict[item['spk_name']]
        sample['spk_id'] = spk_id
        sample['ref_spk_mel'] = ref_spk_mel
        sample['spk_name'] = item['spk_name']
        # sample['emotion'] = item['emotion']
        # sample['sub_dataset'] = item['sub_dataset']
        sample['item_name'] = item['item_name']
        sample['style_embed'] = torch.Tensor(item['style_embed'])
        sample['ph_token'] = torch.LongTensor(item['ph_token'])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(TCTTSEditorDataset, self).collater(samples)
        hparams = self.hparams
        ph_tokens = collate_1d_or_2d([s['ph_token'] for s in samples], 0)
        ref_spk_mels = collate_1d_or_2d([s['ref_spk_mel'] for s in samples], 0.0)
        spk_mel_lengths = torch.LongTensor([s['ref_spk_mel'].shape[0] for s in samples])
        mel2ph = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        style_embed = collate_1d_or_2d([s['style_embed'] for s in samples], 0.0)
        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            uv = collate_1d_or_2d([s['uv'] for s in samples])
            pitch = collate_1d_or_2d([s['pitch'] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        
        item_names = [s['item_name'] for s in samples]
        spk_names = [s['spk_name'] for s in samples]
        spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
        # emotions = [s['emotion'] for s in samples]
        # sub_datasets = [s['sub_dataset'] for s in samples]
        batch.update({
            'item_names': item_names,
            'ph_tokens': ph_tokens,
            'ref_spk_mels': ref_spk_mels,
            'spk_mel_lengths': spk_mel_lengths,
            'spk_names': spk_names,
            # 'emotions': emotions,
            # 'sub_datasets': sub_datasets,
            'mel2ph':mel2ph,
            'f0': f0,
            'uv': uv,
            'pitch': pitch,
            'style_embed':style_embed,
            'spk_ids':spk_ids,
        })
        return batch

class TTCEditorDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix,shuffle,items,data_dir)
        with open(f'./{self.data_dir}/spk_map.json', 'r') as file:
            spk_dict = json.load(file)
        self.spk_dict = spk_dict
        
    def __getitem__(self, index):
        sample = super(TTCEditorDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample['mel']
        T = mel.shape[0]
        max_frames = hparams['max_frames']
        ph_token = item['ph_token']
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T]
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if hparams['pitch_type'] == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)
        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        

        if hparams.get('parallel',False):
            ### 可以考虑使用target的mel，但是提供word-level的pooling，，以及slice部分word的embedding
            if hparams.get('use_crop',False):
                ref_spk_mel = random_crop_with_prob(torch.Tensor(item['mel'])[:max_frames],0.8)
            else:
                ref_spk_mel = torch.Tensor(item['mel'])[:max_frames]
        else:
            ref_spk_mel = torch.Tensor(item['ref_spk_mel'])[:max_frames]
        if hparams.get('use_random',False):
            if random.random() < 0.5:
                ref_spk_mel = torch.Tensor(item['mel'])[:max_frames]
            else:
                ref_spk_mel = torch.Tensor(item['ref_spk_mel'])[:max_frames]
        
        spk_id = self.spk_dict[item['spk_name']]
        sample['spk_id'] = spk_id
        sample['ref_spk_mel'] = ref_spk_mel
        sample['spk_name'] = item['spk_name']
        sample['emotion'] = item['emotion']
        sample['sub_dataset'] = item['sub_dataset']
        sample['item_name'] = item['item_name']
        sample['style_embed'] = torch.Tensor(item['style_embed'])
        sample['ph_token'] = torch.LongTensor(item['ph_token'])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(TTCEditorDataset, self).collater(samples)
        hparams = self.hparams
        ph_tokens = collate_1d_or_2d([s['ph_token'] for s in samples], 0)
        ref_spk_mels = collate_1d_or_2d([s['ref_spk_mel'] for s in samples], 0.0)
        spk_mel_lengths = torch.LongTensor([s['ref_spk_mel'].shape[0] for s in samples])
        mel2ph = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        style_embed = collate_1d_or_2d([s['style_embed'] for s in samples], 0.0)
        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            uv = collate_1d_or_2d([s['uv'] for s in samples])
            pitch = collate_1d_or_2d([s['pitch'] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        
        item_names = [s['item_name'] for s in samples]
        spk_names = [s['spk_name'] for s in samples]
        spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
        emotions = [s['emotion'] for s in samples]
        sub_datasets = [s['sub_dataset'] for s in samples]
        batch.update({
            'item_names': item_names,
            'ph_tokens': ph_tokens,
            'ref_spk_mels': ref_spk_mels,
            'spk_mel_lengths': spk_mel_lengths,
            'spk_names': spk_names,
            'emotions': emotions,
            'sub_datasets': sub_datasets,
            'mel2ph':mel2ph,
            'f0': f0,
            'uv': uv,
            'pitch': pitch,
            'style_embed':style_embed,
            'spk_ids':spk_ids,
        })
        return batch

class WavLMTTCDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(prefix,shuffle,items,data_dir)
        with open(f'./{self.data_dir}/spk_map.json', 'r') as file:
            spk_dict = json.load(file)
        self.spk_dict = spk_dict
        
    def __getitem__(self, index):
        sample = super(WavLMTTCDataset, self).__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams
        mel = sample['mel']
        T = mel.shape[0]
        max_frames = hparams['max_frames']
        ph_token = item['ph_token']
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T]
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            pitch = torch.LongTensor(item.get(hparams.get('pitch_key', 'pitch')))[:T]
            f0, uv = norm_interp_f0(item["f0"][:T])
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
            if hparams['pitch_type'] == 'ph':
                if "f0_ph" in item:
                    f0 = torch.FloatTensor(item['f0_ph'])
                else:
                    f0 = denorm_f0(f0, None)
                f0_phlevel_sum = torch.zeros_like(ph_token).float().scatter_add(0, mel2ph - 1, f0)
                f0_phlevel_num = torch.zeros_like(ph_token).float().scatter_add(
                    0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
                f0_ph = f0_phlevel_sum / f0_phlevel_num
                f0, uv = norm_interp_f0(f0_ph)
        else:
            f0, uv, pitch = None, None, None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        

        if hparams.get('parallel',False):
            ### 可以考虑使用target的mel，但是提供word-level的pooling，，以及slice部分word的embedding
            if hparams.get('use_crop',False):
                ref_spk_mel = random_crop_with_prob(torch.Tensor(item['mel'])[:max_frames],0.8)
            else:
                ref_spk_mel = torch.Tensor(item['mel'])[:max_frames]
        else:
            ref_spk_mel = torch.Tensor(item['ref_spk_mel'])[:max_frames]
        if hparams.get('use_random',False):
            if random.random() < 0.5:
                ref_spk_mel = torch.Tensor(item['mel'])[:max_frames]
            else:
                ref_spk_mel = torch.Tensor(item['ref_spk_mel'])[:max_frames]
        
        spk_id = self.spk_dict[item['spk_name']]
        sample['spk_id'] = spk_id
        sample['ref_spk_mel'] = ref_spk_mel
        sample['spk_name'] = item['spk_name']
        sample['emotion'] = item['emotion']
        sample['sub_dataset'] = item['sub_dataset']
        sample['item_name'] = item['item_name']
        sample['style_embed'] = torch.Tensor(item['style_embed'])
        sample['ph_token'] = torch.LongTensor(item['ph_token'])
        sample['wavlm_spk_embed'] = torch.Tensor(item['wavlm_embed'])
        # sample['codec_embed'] = torch.Tensor(item['codec_embed'])
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(WavLMTTCDataset, self).collater(samples)
        hparams = self.hparams
        ph_tokens = collate_1d_or_2d([s['ph_token'] for s in samples], 0)
        ref_spk_mels = collate_1d_or_2d([s['ref_spk_mel'] for s in samples], 0.0)
        spk_mel_lengths = torch.LongTensor([s['ref_spk_mel'].shape[0] for s in samples])
        mel2ph = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        style_embed = collate_1d_or_2d([s['style_embed'] for s in samples], 0.0)
        wavlm_spk_embed = collate_1d_or_2d([s['wavlm_spk_embed'] for s in samples], 0.0)
        # codec_embed = collate_1d_or_2d([s['codec_embed'] for s in samples], 0.0)
        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            uv = collate_1d_or_2d([s['uv'] for s in samples])
            pitch = collate_1d_or_2d([s['pitch'] for s in samples])
        else:
            f0, uv, pitch = None, None, None
        
        item_names = [s['item_name'] for s in samples]
        spk_names = [s['spk_name'] for s in samples]
        spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
        emotions = [s['emotion'] for s in samples]
        sub_datasets = [s['sub_dataset'] for s in samples]
        batch.update({
            'item_names': item_names,
            'ph_tokens': ph_tokens,
            'ref_spk_mels': ref_spk_mels,
            'spk_mel_lengths': spk_mel_lengths,
            'spk_names': spk_names,
            'emotions': emotions,
            'sub_datasets': sub_datasets,
            'mel2ph':mel2ph,
            'f0': f0,
            'uv': uv,
            'pitch': pitch,
            'style_embed':style_embed,
            'wavlm_spk_embed':wavlm_spk_embed,
            # 'codec_embed':codec_embed,
            'spk_ids':spk_ids,
        })
        return batch

class RecognizerDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        
        with open(f'{self.data_dir}/spk_map.json', 'r') as file:
            spk_dict = json.load(file)
        with open(f'{self.data_dir}/emotion_map.json', 'r') as file:
            emo_dict = json.load(file)    
        self.spk_dict = spk_dict
        self.emo_dict = emo_dict
        
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == 'train' and hparams['min_frames'] > 0:
                self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        assert len(item['mel']) == self.sizes[index], (len(item['mel']), self.sizes[index])
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]
        
        emotion_id = self.emo_dict[item['emotion']]
        spk_id = self.spk_dict[item['spk_name']]
        
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
            "emotion":emotion_id,
            "spk":spk_id,
        }
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0)
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])
        emotion = torch.LongTensor([s['emotion'] for s in samples])
        spk = torch.LongTensor([s['spk'] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'mels': mels,
            'mel_lengths': mel_lengths,
            'emotion': emotion,
            'spk':spk,
        }
        return batch


class Emotion2VecDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.hparams = hparams
        self.indexed_ds = None
        
        with open(f'{self.data_dir}/spk_map.json', 'r') as file:
            spk_dict = json.load(file)
        with open(f'{self.data_dir}/emotion_map.json', 'r') as file:
            emo_dict = json.load(file)    
        self.spk_dict = spk_dict
        self.emo_dict = emo_dict
        
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == 'train' and hparams['min_frames'] > 0:
                self.avail_idxs = [x for x in self.avail_idxs if self.sizes[x] >= hparams['min_frames']]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        
        emotion_id = self.emo_dict[item['emotion']]
        spk_id = self.spk_dict[item['spk_name']]
        emo_embed = torch.Tensor(item['emo_embed'])
        
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "emotion":emotion_id,
            "spk":spk_id,
            "emo_embed":emo_embed,
        }
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        emotion = torch.LongTensor([s['emotion'] for s in samples])
        spk = torch.LongTensor([s['spk'] for s in samples])
        emo_embeds = collate_1d_or_2d([s['emo_embed'] for s in samples], 0.0)

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'emotion': emotion,
            'spk':spk,
            'emo_embeds':emo_embeds,
        }
        return batch