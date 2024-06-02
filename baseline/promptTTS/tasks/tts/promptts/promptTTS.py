import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils.commons.hparams import hparams

from tasks.tts.fs import FastSpeechTask
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls, BaseVocoder
from modules.tts.promptTTS.promptts import PromptTTS
from utils.audio.pitch.utils import denorm_f0
from tasks.tts.promptts.promptts_dataset_utils import PromptTTSDataset
from utils.audio.align import mel2token_to_dur


class PromptTTSTask(FastSpeechTask):
    def __init__(self):
        super(PromptTTSTask, self).__init__()
        self.dataset_cls = PromptTTSDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = PromptTTS(dict_size,hparams)
                    

    def run_model(self, sample, infer=False):
        mel = sample['mels']  # [B, T_s, 80]
        ph_token = sample['ph_tokens']
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        style_embed = sample['style_embed']
        
        if not infer:
            output = self.model(ph_token,style_embed,mel2ph,f0,uv,infer=infer,global_step=self.global_step)
            losses = {}
            self.add_mel_loss(output['mel_out'], mel, losses)
            self.add_dur_loss(output['dur'], mel2ph, ph_token, losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            mel2ph,f0,uv = None,None,None
            global_step=10000000
            output = self.model(ph_token,style_embed,mel2ph,f0,uv,infer=infer,global_step=global_step)
            return output
   

    def validation_step(self, sample, batch_idx):
        outputs = {}
        mel = sample['mels']  # [B, T_s, 80]
        ph_token = sample['ph_tokens']
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        style_embed = sample['style_embed']

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.commons.tensor_utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(ph_token,style_embed,mel2ph,f0,uv,infer=True)
            gt_f0 = denorm_f0(f0, uv, pitch_padding=(mel2ph == 0))
            f0 = model_out['f0_denorm_pred']
                
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'],name=f'mel_val_{batch_idx}')
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], gt_f0, f0, is_mel=True)
            
        return outputs

    def _training_step(self, sample, batch_idx, optimizer_idx):       
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output
    
    def plot_wav(self, batch_idx, gt_wav, wav_out, gt_f0, f0, is_mel=False, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = gt_f0[0].cpu().numpy()
        f0 = f0[0].cpu().numpy()
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav, gt_f0)
            wav_out = self.vocoder.spec2wav(wav_out, f0)
        self.logger.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        self.logger.add_audio(f'wav_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
