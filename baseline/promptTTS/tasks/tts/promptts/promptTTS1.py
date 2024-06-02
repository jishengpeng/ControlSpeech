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
from modules.tts.ps_adv.multi_window_disc import Discriminator
from utils.nn.schedulers import RSQRTSchedule, NoneSchedule, WarmupSchedule


class PromptTTSTask(FastSpeechTask):
    def __init__(self):
        super(PromptTTSTask, self).__init__()
        self.dataset_cls = PromptTTSDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self.mse_loss_fn = torch.nn.MSELoss()    
        self.build_disc_model()

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = PromptTTS(dict_size,hparams)
        self.gen_params = []
        for name, param in self.model.named_parameters():
            self.gen_params.append(param)
    
    def build_disc_model(self):
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())
                    

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

    # def _training_step(self, sample, batch_idx, optimizer_idx):       
    #     loss_output, _ = self.run_model(sample)
    #     total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
    #     loss_output['batch_size'] = sample['nsamples']
    #     return total_loss, loss_output
    def _training_step(self, sample, batch_idx, optimizer_idx):
        loss_output = {}
        loss_weights = {}
        disc_start = self.global_step >= hparams["disc_start_steps"] and hparams['lambda_mel_adv'] > 0
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            loss_output, model_out = self.run_model(sample, infer=False)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                mel_p = model_out['mel_out']
                if hasattr(self.model, 'out2mel'):
                    mel_p = self.model.out2mel(mel_p)
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output['a'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    loss_weights['a'] = hparams['lambda_mel_adv']
                if pc_ is not None:
                    loss_output['ac'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
                    loss_weights['ac'] = hparams['lambda_mel_adv']
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                model_out = self.model_out_gt
                mel_g = sample['mels']
                mel_p = model_out['mel_out']
                o = self.mel_disc(mel_g)
                p, pc = o['y'], o['y_c']
                o_ = self.mel_disc(mel_p)
                p_, pc_ = o_['y'], o_['y_c']
                if p_ is not None:
                    loss_output["r"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                    loss_output["f"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
                if pc_ is not None:
                    loss_output["rc"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
                    loss_output["fc"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
            if len(loss_output) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output
    
    def build_optimizer(self, model):      
        optimizer_gen = torch.optim.AdamW(self.gen_params,lr=hparams['lr'])
        optimizer_disc = torch.optim.AdamW(self.disc_params,lr=hparams['disc_lr'])
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return [
            FastSpeechTask.build_scheduler(self, optimizer[0]), # Generator Scheduler
            WarmupSchedule(optimizer[1], hparams['disc_lr'], hparams['warmup_updates'])
        ]

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.gen_params, hparams['clip_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        assert self.scheduler is not None,'scheduler is None,error!!!!'
        self.scheduler[0].step(self.global_step // hparams['accumulate_grad_batches'])
        self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])
    
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
