import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils.commons.hparams import hparams

from tasks.tts.fs import FastSpeechTask
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls, BaseVocoder
from modules.tts.ei_vc.ei_netowrk import EI_Network
from modules.tts.vc_clap.diffvc_clap import Diffusion_VC_CLAP,NS2_VC_CLAP,NS2_TTS_CLAP,\
                                            NS2_TTS_Diff,Codec_WavLM_TTS_Diff
from utils.audio.pitch.utils import denorm_f0
from tasks.tts.vc_editor_dataset_utils import VCCLAPEditorDataset,TCTTSEditorDataset,WavLMTTCDataset
from modules.tts.ProDiff.model.diff.net import DiffNet
from modules.tts.vc_clap.naturalspeech2 import NSDiffNet
from utils.audio.align import mel2token_to_dur

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
    'NSDiffNet': lambda hp: NSDiffNet(dim=256,depth=2,dim_prompt=hp['hidden_size'],
                                        cond_drop_prob=0.25,condition_on_prompt=True,
                                        num_latents_m=hp['sbank_size'])
}

class VCCLAPEditorSpeechTask(FastSpeechTask):
    def __init__(self):
        super(VCCLAPEditorSpeechTask, self).__init__()
        self.dataset_cls = VCCLAPEditorDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self.criterion = nn.L1Loss(reduction='sum')
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = Diffusion_VC_CLAP(
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max']
        )
                    

    def run_model(self, sample, infer=False):
        mel = sample['mels']  # [B, T_s, 80]
        ref_spk_mel = sample['ref_spk_mels']
        ph_token = sample['ph_tokens']
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        style_embed = sample['style_embed']
        
        if not infer:
            output = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,infer)
            losses = {}
            self.add_mel_loss(output['mel_out'], mel, losses)
            self.add_dur_loss(output['dur'], mel2ph, ph_token, losses=losses)
            self.add_cycle_consistency_loss(output,sample,losses=losses)
            # self.add_grl_loss(output,sample,losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            mel2ph,f0,uv = None,None,None
            output = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,infer)
            return output
        
    def add_dur_loss(self, dur_pred, mel2ph, ph_token, losses=None):
        """

        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param ph_token: [B, T]
        :param losses:
        :return:
        """
        B, T = ph_token.shape
        nonpadding = (ph_token != 100).float()
        dur_gt = mel2token_to_dur(mel2ph, T).float() * nonpadding
        losses['pdur'] = F.mse_loss((dur_pred + 1).log(), (dur_gt + 1).log(), reduction='none')
        losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
        losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']
        # use linear scale for sentence
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']
    
    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float() if hparams['pitch_type'] == 'frame' \
            else (sample['ph_tokens'] != 100).float()
        p_pred = output['pitch_pred']
        assert p_pred[..., 0].shape == f0.shape
        if hparams['use_uv'] and hparams['pitch_type'] == 'frame':
            assert p_pred[..., 1].shape == uv.shape, (p_pred.shape, uv.shape)
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = nonpadding * (uv == 0).float()
        f0_pred = p_pred[:, :, 0]
        losses['f0'] = (F.l1_loss(f0_pred, f0, reduction='none') * nonpadding).sum() \
                       / nonpadding.sum() * hparams['lambda_f0']
    
    def add_cycle_consistency_loss(self,output,sample,losses):
        ret = {}
        gt_spk_embed = output['spk_embed']
        self.model.vc_clap_encoder.forward_spk_embed(output['mel_out'], None, ret)
        gen_spk_embed = ret['spk_embed']
        spk_cycle_loss = self.criterion(gt_spk_embed,gen_spk_embed)
        losses['spk_cycle_loss']= hparams.get('lambda_spk_cycle_loss',1.0)*spk_cycle_loss
    
    def add_grl_loss(self,output,sample,losses):
        predit_out = output['predict_out']
        spk_ids = sample['spk_ids']
        spk_grl_loss = self.cross_entropy(predit_out,spk_ids)
        losses['spk_grl_loss'] = spk_grl_loss*hparams.get('spk_grl_loss',0.1)
        

    def validation_step(self, sample, batch_idx):
        outputs = {}
        mel = sample['mels']  # [B, T_s, 80]
        ref_spk_mel = sample['ref_spk_mels']
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
        ## 音色转换、情绪转换、两者都转换
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,False)
                
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'],name=f'mel_val_{batch_idx}')
            ## source wav, drive wav
            self.plot_wav(batch_idx, sample['mels'], f'gt_{batch_idx}', is_mel=True)
            self.plot_wav(batch_idx, model_out['mel_out'],f'wav_{batch_idx}', is_mel=True)
            self.plot_wav(batch_idx, sample['ref_spk_mels'],f'wav_{batch_idx}_ref', is_mel=True)
            
        return outputs
    
    def test_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return:
        """
        assert sample['mels'].shape[0] == 1, 'only support batch_size=1 in inference'
        import random
        from utils.audio.io import save_wav
        random_number = random.randint(0, 2)
        if batch_idx >=20:
            return {}
        
        save_dir = f'./inference/infer_out/{hparams["exp_name"]}/{batch_idx}'
        os.makedirs(save_dir,exist_ok=True)
        
        model_out = self.run_model(sample, infer=True)
        
        mel = sample['mels'][0].cpu().numpy()
        spk_mel = sample['ref_spk_mels'][0].cpu().numpy()
        gen_mel = model_out['mel_out'][0].cpu().numpy()

        wav = self.vocoder.spec2wav(mel)
        spk_wav = self.vocoder.spec2wav(spk_mel)
        gen_wav = self.vocoder.spec2wav(gen_mel)
        
        
        self.spec_to_figure(gen_mel,vmin=hparams['mel_vmin'],vmax=hparams['mel_vmax'],save_path=f'{save_dir}/gem_mel.png')
        save_wav(wav,f'{save_dir}/gt.wav',sr=16000)
        save_wav(spk_wav,f'{save_dir}/ref_spk.wav',sr=16000)
        save_wav(gen_wav,f'{save_dir}/gen.wav',sr=16000)
        
        return {}

    def spec_to_figure(self,spec, vmin=None, vmax=None, title='', f0s=None, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        LINE_COLORS = ['w', 'r', 'orange', 'k', 'cyan', 'm', 'b', 'lime', 'g', 'brown', 'navy']
        
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        H = spec.shape[1] // 2
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        if f0s is not None:
            ax = plt.gca()
            ax2 = ax.twinx()
            if not isinstance(f0s, dict):
                f0s = {'f0': f0s}
            for i, (k, f0) in enumerate(f0s.items()):
                if isinstance(f0, torch.Tensor):
                    f0 = f0.cpu().numpy()
                ax2.plot(f0, label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.5)
            ax2.set_ylim(0, 1000)
            ax2.legend()
        fig.savefig(save_path)

    def _training_step(self, sample, batch_idx, optimizer_idx):       
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output
    
    # def on_train_end(self):
    #     if hparams['start_update_recognizer_steps']==self.global_step:
    #         for k,v in self.timbre_disc.named_parameters():
    #             v.requires_grad=True
    #         for k,v in self.emo_disc.named_parameters():
    #             v.requires_grad=True

    ############
    # validation plots
    ############
    # def plot_wav(self, batch_idx, gt_wav, wav_out, gt_label,pred_label, is_mel=False, name=None):
    #     gt_wav = gt_wav[0].cpu().numpy()
    #     wav_out = wav_out[0].cpu().numpy()
    #     if is_mel:
    #         gt_wav = self.vocoder.spec2wav(gt_wav)
    #         wav_out = self.vocoder.spec2wav(wav_out)
    #     self.logger.add_audio(gt_label, gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
    #     self.logger.add_audio(pred_label, wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        
    def plot_wav(self, batch_idx, wav, label, is_mel=False, name=None):
        wav = wav[0].cpu().numpy()
        if is_mel:
            wav_out = self.vocoder.spec2wav(wav)
        self.logger.add_audio(label, wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        
class NS2VCCLAPEditorSpeechTask(FastSpeechTask):
    def __init__(self):
        super(NS2VCCLAPEditorSpeechTask, self).__init__()
        self.dataset_cls = VCCLAPEditorDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self.criterion = nn.L1Loss(reduction='sum')
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = NS2_VC_CLAP(
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max']
        )
                    

    def run_model(self, sample, infer=False):
        mel = sample['mels']  # [B, T_s, 80]
        ref_spk_mel = sample['ref_spk_mels']
        ph_token = sample['ph_tokens']
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        style_embed = sample['style_embed']
        
        if not infer:
            output = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,infer)
            losses = {}
            self.add_mel_loss(output['mel_out'], mel, losses)
            self.add_dur_loss(output['dur'], mel2ph, ph_token, losses=losses)
            # self.add_cycle_consistency_loss(output,sample,losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            mel2ph,f0,uv = None,None,None
            output = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,infer)
            return output
        
    def add_dur_loss(self, dur_pred, mel2ph, ph_token, losses=None):
        """

        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param ph_token: [B, T]
        :param losses:
        :return:
        """
        B, T = ph_token.shape
        nonpadding = (ph_token != 100).float()
        dur_gt = mel2token_to_dur(mel2ph, T).float() * nonpadding
        losses['pdur'] = F.mse_loss((dur_pred + 1).log(), (dur_gt + 1).log(), reduction='none')
        losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
        losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']
        # use linear scale for sentence
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']
    
    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float() if hparams['pitch_type'] == 'frame' \
            else (sample['ph_tokens'] != 100).float()
        p_pred = output['pitch_pred']
        assert p_pred[..., 0].shape == f0.shape
        if hparams['use_uv'] and hparams['pitch_type'] == 'frame':
            assert p_pred[..., 1].shape == uv.shape, (p_pred.shape, uv.shape)
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                           / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = nonpadding * (uv == 0).float()
        f0_pred = p_pred[:, :, 0]
        losses['f0'] = (F.l1_loss(f0_pred, f0, reduction='none') * nonpadding).sum() \
                       / nonpadding.sum() * hparams['lambda_f0']
    
    def add_cycle_consistency_loss(self,output,sample,losses):
        ret = {}
        gt_spk_embed = output['spk_embed']
        self.model.vc_clap_encoder.forward_spk_embed(output['mel_out'], None, ret)
        gen_spk_embed = ret['spk_embed']
        spk_cycle_loss = self.criterion(gt_spk_embed,gen_spk_embed)
        losses['spk_cycle_loss']= hparams.get('lambda_spk_cycle_loss',1.0)*spk_cycle_loss
   

    def validation_step(self, sample, batch_idx):
        outputs = {}
        mel = sample['mels']  # [B, T_s, 80]
        ref_spk_mel = sample['ref_spk_mels']
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
        ## 音色转换、情绪转换、两者都转换
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,False)
                
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'],name=f'mel_val_{batch_idx}')
            ## source wav, drive wav
            self.plot_wav(batch_idx, sample['mels'], f'gt_{batch_idx}', is_mel=True)
            self.plot_wav(batch_idx, model_out['mel_out'],f'wav_{batch_idx}', is_mel=True)
            self.plot_wav(batch_idx, sample['ref_spk_mels'],f'wav_{batch_idx}_ref', is_mel=True)
            
        return outputs
    
    def test_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return:
        """
        assert sample['mels'].shape[0] == 1, 'only support batch_size=1 in inference'
        import random
        from utils.audio.io import save_wav
        random_number = random.randint(0, 2)
        if batch_idx >=20:
            return {}
        
        save_dir = f'./inference/infer_out/{hparams["exp_name"]}/{batch_idx}'
        os.makedirs(save_dir,exist_ok=True)
        
        model_out = self.run_model(sample, infer=True)
        
        mel = sample['mels'][0].cpu().numpy()
        spk_mel = sample['ref_spk_mels'][0].cpu().numpy()
        gen_mel = model_out['mel_out'][0].cpu().numpy()

        wav = self.vocoder.spec2wav(mel)
        spk_wav = self.vocoder.spec2wav(spk_mel)
        gen_wav = self.vocoder.spec2wav(gen_mel)
        
        
        self.spec_to_figure(gen_mel,vmin=hparams['mel_vmin'],vmax=hparams['mel_vmax'],save_path=f'{save_dir}/gem_mel.png')
        save_wav(wav,f'{save_dir}/gt.wav',sr=16000)
        save_wav(spk_wav,f'{save_dir}/ref_spk.wav',sr=16000)
        save_wav(gen_wav,f'{save_dir}/gen.wav',sr=16000)
        
        return {}

    def spec_to_figure(self,spec, vmin=None, vmax=None, title='', f0s=None, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        LINE_COLORS = ['w', 'r', 'orange', 'k', 'cyan', 'm', 'b', 'lime', 'g', 'brown', 'navy']
        
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        H = spec.shape[1] // 2
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        if f0s is not None:
            ax = plt.gca()
            ax2 = ax.twinx()
            if not isinstance(f0s, dict):
                f0s = {'f0': f0s}
            for i, (k, f0) in enumerate(f0s.items()):
                if isinstance(f0, torch.Tensor):
                    f0 = f0.cpu().numpy()
                ax2.plot(f0, label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.5)
            ax2.set_ylim(0, 1000)
            ax2.legend()
        fig.savefig(save_path)

    def _training_step(self, sample, batch_idx, optimizer_idx):       
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output
    
    # def on_train_end(self):
    #     if hparams['start_update_recognizer_steps']==self.global_step:
    #         for k,v in self.timbre_disc.named_parameters():
    #             v.requires_grad=True
    #         for k,v in self.emo_disc.named_parameters():
    #             v.requires_grad=True

    ############
    # validation plots
    ############
    # def plot_wav(self, batch_idx, gt_wav, wav_out, gt_label,pred_label, is_mel=False, name=None):
    #     gt_wav = gt_wav[0].cpu().numpy()
    #     wav_out = wav_out[0].cpu().numpy()
    #     if is_mel:
    #         gt_wav = self.vocoder.spec2wav(gt_wav)
    #         wav_out = self.vocoder.spec2wav(wav_out)
    #     self.logger.add_audio(gt_label, gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
    #     self.logger.add_audio(pred_label, wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        
    def plot_wav(self, batch_idx, wav, label, is_mel=False, name=None):
        wav = wav[0].cpu().numpy()
        if is_mel:
            wav_out = self.vocoder.spec2wav(wav)
        self.logger.add_audio(label, wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)

class NS2TTSCLAPEditorSpeechTask(FastSpeechTask):
    def __init__(self):
        super(NS2TTSCLAPEditorSpeechTask, self).__init__()
        self.dataset_cls = TCTTSEditorDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self.criterion = nn.L1Loss(reduction='sum')
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.spk_cnt = None

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = NS2_TTS_CLAP(
            phone_encoder=self.token_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max']
        )
                    

    def run_model(self, sample, infer=False):
        mel = sample['mels']  # [B, T_s, 80]
        ref_spk_mel = sample['ref_spk_mels']
        ph_token = sample['ph_tokens']
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        style_embed = sample['style_embed']
        
        if not infer:
            output = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,infer,global_step=self.global_step)
            losses = {}
            self.add_mel_loss(output['mel_out'], mel, losses)
            ### 用fs的对应loss
            self.add_dur_loss(output['dur'], mel2ph, ph_token, losses=losses)
            # self.add_cycle_consistency_loss(output,sample,losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            mel2ph,f0,uv = None,None,None
            global_step=200000
            output = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,infer,global_step=global_step)
            return output
    
    def add_cycle_consistency_loss(self,output,sample,losses):
        ret = {}
        gt_spk_embed = output['spk_embed']
        self.model.vc_clap_encoder.forward_spk_embed(output['mel_out'], None, ret)
        gen_spk_embed = ret['spk_embed']
        spk_cycle_loss = self.criterion(gt_spk_embed,gen_spk_embed)
        losses['spk_cycle_loss']= hparams.get('lambda_spk_cycle_loss',1.0)*spk_cycle_loss
   

    def validation_step(self, sample, batch_idx):
        outputs = {}
        mel = sample['mels']  # [B, T_s, 80]
        ref_spk_mel = sample['ref_spk_mels']
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
            model_out = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,True)
            gt_f0 = denorm_f0(f0, uv, pitch_padding=(mel2ph == 0))
            f0 = model_out['f0_denorm_pred']
                
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'],name=f'mel_val_{batch_idx}')
            ## source wav, drive wav
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], gt_f0, f0, is_mel=True)
            # self.plot_wav(batch_idx, sample['mels'], f'gt_{batch_idx}', is_mel=True)
            # self.plot_wav(batch_idx, model_out['mel_out'],f'wav_{batch_idx}', is_mel=True)
            # self.plot_wav(batch_idx, sample['ref_spk_mels'],f'wav_{batch_idx}_ref', is_mel=True)
            
        return outputs
    
    def test_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return:
        """
        assert sample['mels'].shape[0] == 1, 'only support batch_size=1 in inference'
        import random
        from utils.audio.io import save_wav

        self.spk_names = ['MEAD-W018','CREMA-D-1028','ESD-0011','ESD-0016','MESS-F3','MESS-M3',
                'RAVDESS-05','TESS-OAF','RAVDESS-10','MEAD-M003']
        if self.spk_cnt==None:
            self.spk_cnt = {}
            for name in self.spk_names:
                self.spk_cnt[name] = 20
        cur_spk_name = sample['spk_names'][0]
        emotion = sample['emotions'][0]
        if cur_spk_name in self.spk_names and self.spk_cnt[cur_spk_name]>0 and emotion=='neutral':
            model_out = self.run_model(sample, infer=True)
            spk_embed = model_out['spk_embed'].detach().cpu().numpy()
            print(spk_embed.shape)
            self.spk_cnt[cur_spk_name] = self.spk_cnt[cur_spk_name]-1
            return {'spk_name':cur_spk_name,'spk_embed':spk_embed,'emotion':emotion}
            

        
        # save_dir = f'./inference/infer_out/{hparams["exp_name"]}/{batch_idx}'
        # os.makedirs(save_dir,exist_ok=True)
        
        # model_out = self.run_model(sample, infer=True)
        # spk_embed = model_out['spk_embed'].detach().cpu().numpy()
        
        # mel = sample['mels'][0].cpu().numpy()
        # spk_mel = sample['ref_spk_mels'][0].cpu().numpy()
        # gen_mel = model_out['mel_out'][0].cpu().numpy()

        # wav = self.vocoder.spec2wav(mel)
        # spk_wav = self.vocoder.spec2wav(spk_mel)
        # gen_wav = self.vocoder.spec2wav(gen_mel)
        
        
        # self.spec_to_figure(gen_mel,vmin=hparams['mel_vmin'],vmax=hparams['mel_vmax'],save_path=f'{save_dir}/gem_mel.png')
        # save_wav(wav,f'{save_dir}/gt.wav',sr=16000)
        # save_wav(spk_wav,f'{save_dir}/ref_spk.wav',sr=16000)
        # save_wav(gen_wav,f'{save_dir}/gen.wav',sr=16000)
        return None
    
    def test_end(self, outputs):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        # 假设你有一个名为 embeddings 的数组，其中包含你的 embedding 数据
        # 假设 labels 是对应的标签
        # 请替换为你自己的实际数据
        labels =[]
        emotion_labels = []
        spk_embed_list = []
        name_id_map = {}
        emotions = ["surprised", "contempt", "sad", "neutral", "disgusted", "happy", "angry", "fear"]
        emotion_id_map = {emo:i  for i,emo in enumerate(emotions)}
        for i,spk_name in enumerate(self.spk_names):
            name_id_map[spk_name]=i
        for out in outputs:
            if out!=None:
                spk_embed_list.append(out['spk_embed'])
                labels.append(name_id_map[out['spk_name']])
                emotion_labels.append(emotion_id_map[out['emotion']])
        nums = len(self.spk_names)
        labels = np.array(labels)
        # nums = len(emotion_id_map)
        # labels = np.array(emotion_labels)

        embeddings = np.concatenate(spk_embed_list,axis=0)

        # 使用 t-SNE 对 embedding 进行降维
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(embeddings)

        # 绘制聚类图
        plt.figure(figsize=(8, 6))

        # 假设你有 10 个类别
        for i in range(nums):
            indices = labels == i
            plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=f'{self.spk_names[i]}')
            # plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=f'{emotions[i]}')

        plt.title('t-SNE Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        save_dir = f'/home/ywx1252669/VC_CLAP/inference/infer_out/{hparams["exp_name"]}'
        os.makedirs(save_dir,exist_ok=True)
        plt.savefig(f'{save_dir}/spk_embedddings_emotion.png')
        plt.show()

        return {}

    def spec_to_figure(self,spec, vmin=None, vmax=None, title='', f0s=None, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        LINE_COLORS = ['w', 'r', 'orange', 'k', 'cyan', 'm', 'b', 'lime', 'g', 'brown', 'navy']
        
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        H = spec.shape[1] // 2
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        if f0s is not None:
            ax = plt.gca()
            ax2 = ax.twinx()
            if not isinstance(f0s, dict):
                f0s = {'f0': f0s}
            for i, (k, f0) in enumerate(f0s.items()):
                if isinstance(f0, torch.Tensor):
                    f0 = f0.cpu().numpy()
                ax2.plot(f0, label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.5)
            ax2.set_ylim(0, 1000)
            ax2.legend()
        fig.savefig(save_path)

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
    
    # def plot_wav(self, batch_idx, wav, label, is_mel=False, name=None):
    #     wav = wav[0].cpu().numpy()
    #     if is_mel:
    #         wav_out = self.vocoder.spec2wav(wav)
    #     self.logger.add_audio(label, wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)

class NS2TTSDiffEditorSpeechTask(FastSpeechTask):
    def __init__(self):
        super(NS2TTSDiffEditorSpeechTask, self).__init__()
        self.dataset_cls = TCTTSEditorDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self.criterion = nn.L1Loss(reduction='sum')
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.spk_cnt = None

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = NS2_TTS_Diff(
            phone_encoder=self.token_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max']
        )
                    

    def run_model(self, sample, infer=False):
        mel = sample['mels']  # [B, T_s, 80]
        ref_spk_mel = sample['ref_spk_mels']
        ph_token = sample['ph_tokens']
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        style_embed = sample['style_embed']
        
        if not infer:
            output = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,infer,global_step=self.global_step)
            losses = {}
            self.add_mel_loss(output['mel_out'], mel, losses)
            ### 用fs的对应loss
            self.add_dur_loss(output['dur'], mel2ph, ph_token, losses=losses)
            # self.add_cycle_consistency_loss(output,sample,losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            mel2ph,f0,uv = None,None,None
            global_step=200000
            output = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,infer,global_step=global_step)
            return output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        mel = sample['mels']  # [B, T_s, 80]
        ref_spk_mel = sample['ref_spk_mels']
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
            model_out = self.model(mel,ref_spk_mel,ph_token,style_embed,mel2ph,f0,uv,True)
                
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'],name=f'mel_val_{batch_idx}')
            ## source wav, drive wav
            self.plot_wav(batch_idx, sample['mels'], f'gt_{batch_idx}', is_mel=True)
            self.plot_wav(batch_idx, model_out['mel_out'],f'wav_{batch_idx}', is_mel=True)
            self.plot_wav(batch_idx, sample['ref_spk_mels'],f'wav_{batch_idx}_ref', is_mel=True)
            
        return outputs
    
    def test_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return:
        """
        assert sample['mels'].shape[0] == 1, 'only support batch_size=1 in inference'
        import random
        from utils.audio.io import save_wav

        self.spk_names = ['MEAD-W018','CREMA-D-1028','ESD-0011','ESD-0016','MESS-F3','MESS-M3',
                'RAVDESS-05','TESS-OAF','RAVDESS-10','MEAD-M003']
        if self.spk_cnt==None:
            self.spk_cnt = {}
            for name in self.spk_names:
                self.spk_cnt[name] = 20
        cur_spk_name = sample['spk_names'][0]
        emotion = sample['emotions'][0]
        if cur_spk_name in self.spk_names and self.spk_cnt[cur_spk_name]>0 and emotion=='neutral':
            model_out = self.run_model(sample, infer=True)
            spk_embed = model_out['spk_embed'].detach().cpu().numpy()
            print(spk_embed.shape)
            self.spk_cnt[cur_spk_name] = self.spk_cnt[cur_spk_name]-1
            return {'spk_name':cur_spk_name,'spk_embed':spk_embed,'emotion':emotion}
            

        
        # save_dir = f'./inference/infer_out/{hparams["exp_name"]}/{batch_idx}'
        # os.makedirs(save_dir,exist_ok=True)
        
        # model_out = self.run_model(sample, infer=True)
        # spk_embed = model_out['spk_embed'].detach().cpu().numpy()
        
        # mel = sample['mels'][0].cpu().numpy()
        # spk_mel = sample['ref_spk_mels'][0].cpu().numpy()
        # gen_mel = model_out['mel_out'][0].cpu().numpy()

        # wav = self.vocoder.spec2wav(mel)
        # spk_wav = self.vocoder.spec2wav(spk_mel)
        # gen_wav = self.vocoder.spec2wav(gen_mel)
        
        
        # self.spec_to_figure(gen_mel,vmin=hparams['mel_vmin'],vmax=hparams['mel_vmax'],save_path=f'{save_dir}/gem_mel.png')
        # save_wav(wav,f'{save_dir}/gt.wav',sr=16000)
        # save_wav(spk_wav,f'{save_dir}/ref_spk.wav',sr=16000)
        # save_wav(gen_wav,f'{save_dir}/gen.wav',sr=16000)
        return None
    
    def test_end(self, outputs):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        # 假设你有一个名为 embeddings 的数组，其中包含你的 embedding 数据
        # 假设 labels 是对应的标签
        # 请替换为你自己的实际数据
        labels =[]
        emotion_labels = []
        spk_embed_list = []
        name_id_map = {}
        emotions = ["surprised", "contempt", "sad", "neutral", "disgusted", "happy", "angry", "fear"]
        emotion_id_map = {emo:i  for i,emo in enumerate(emotions)}
        for i,spk_name in enumerate(self.spk_names):
            name_id_map[spk_name]=i
        for out in outputs:
            if out!=None:
                spk_embed_list.append(out['spk_embed'])
                labels.append(name_id_map[out['spk_name']])
                emotion_labels.append(emotion_id_map[out['emotion']])
        nums = len(self.spk_names)
        labels = np.array(labels)
        # nums = len(emotion_id_map)
        # labels = np.array(emotion_labels)

        embeddings = np.concatenate(spk_embed_list,axis=0)

        # 使用 t-SNE 对 embedding 进行降维
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(embeddings)

        # 绘制聚类图
        plt.figure(figsize=(8, 6))

        # 假设你有 10 个类别
        for i in range(nums):
            indices = labels == i
            plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=f'{self.spk_names[i]}')
            # plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=f'{emotions[i]}')

        plt.title('t-SNE Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        save_dir = f'/home/ywx1252669/VC_CLAP/inference/infer_out/{hparams["exp_name"]}'
        os.makedirs(save_dir,exist_ok=True)
        plt.savefig(f'{save_dir}/spk_embedddings_emotion.png')
        plt.show()

        return {}

    def spec_to_figure(self,spec, vmin=None, vmax=None, title='', f0s=None, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        LINE_COLORS = ['w', 'r', 'orange', 'k', 'cyan', 'm', 'b', 'lime', 'g', 'brown', 'navy']
        
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        H = spec.shape[1] // 2
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        if f0s is not None:
            ax = plt.gca()
            ax2 = ax.twinx()
            if not isinstance(f0s, dict):
                f0s = {'f0': f0s}
            for i, (k, f0) in enumerate(f0s.items()):
                if isinstance(f0, torch.Tensor):
                    f0 = f0.cpu().numpy()
                ax2.plot(f0, label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.5)
            ax2.set_ylim(0, 1000)
            ax2.legend()
        fig.savefig(save_path)

    def _training_step(self, sample, batch_idx, optimizer_idx):       
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output
        
    def plot_wav(self, batch_idx, wav, label, is_mel=False, name=None):
        wav = wav[0].cpu().numpy()
        if is_mel:
            wav_out = self.vocoder.spec2wav(wav)
        self.logger.add_audio(label, wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)

class WavLMTTCSpeechTask(FastSpeechTask):
    def __init__(self):
        super(WavLMTTCSpeechTask, self).__init__()
        self.dataset_cls = WavLMTTCDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self.criterion = nn.L1Loss(reduction='sum')
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.spk_cnt = None

    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = Codec_WavLM_TTS_Diff(
            phone_encoder=self.token_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max']
        )
                    

    def run_model(self, sample, infer=False):
        mel = sample['mels']  # [B, T_s, 80]
        ref_spk_mel = sample['ref_spk_mels']
        ph_token = sample['ph_tokens']
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        style_embed = sample['style_embed']
        wavlm_spk_embed = sample['wavlm_spk_embed']
        
        if not infer:
            output = self.model(mel,ref_spk_mel,ph_token,style_embed,wavlm_spk_embed,mel2ph,f0,uv,infer,global_step=self.global_step)
            losses = {}
            self.add_mel_loss(output['mel_out'], mel, losses)
            ### 用fs的对应loss
            self.add_dur_loss(output['dur'], mel2ph, ph_token, losses=losses)
            # self.add_cycle_consistency_loss(output,sample,losses=losses)
            if hparams['use_pitch_embed']:
                self.add_pitch_loss(output, sample, losses)
            return losses, output
        else:
            mel2ph,f0,uv = None,None,None
            global_step=200000
            output = self.model(mel,ref_spk_mel,ph_token,style_embed,wavlm_spk_embed,mel2ph,f0,uv,infer,global_step=global_step)
            return output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        mel = sample['mels']  # [B, T_s, 80]
        ref_spk_mel = sample['ref_spk_mels']
        ph_token = sample['ph_tokens']
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        style_embed = sample['style_embed']
        wavlm_spk_embed = sample['wavlm_spk_embed']

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.commons.tensor_utils.tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(mel,ref_spk_mel,ph_token,style_embed,wavlm_spk_embed,mel2ph,f0,uv,True)
                
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'],name=f'mel_val_{batch_idx}')
            ## source wav, drive wav
            self.plot_wav(batch_idx, sample['mels'], f'gt_{batch_idx}', is_mel=True)
            self.plot_wav(batch_idx, model_out['mel_out'],f'wav_{batch_idx}', is_mel=True)
            self.plot_wav(batch_idx, sample['ref_spk_mels'],f'wav_{batch_idx}_ref', is_mel=True)
            
        return outputs
    
    def test_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return:
        """
        assert sample['mels'].shape[0] == 1, 'only support batch_size=1 in inference'
        import random
        from utils.audio.io import save_wav

        self.spk_names = ['MEAD-W018','CREMA-D-1028','ESD-0011','ESD-0016','MESS-F3','MESS-M3',
                'RAVDESS-05','TESS-OAF','RAVDESS-10','MEAD-M003']
        if self.spk_cnt==None:
            self.spk_cnt = {}
            for name in self.spk_names:
                self.spk_cnt[name] = 20
        cur_spk_name = sample['spk_names'][0]
        emotion = sample['emotions'][0]
        if cur_spk_name in self.spk_names and self.spk_cnt[cur_spk_name]>0 and emotion=='neutral':
            model_out = self.run_model(sample, infer=True)
            spk_embed = model_out['spk_embed'].detach().cpu().numpy()
            print(spk_embed.shape)
            self.spk_cnt[cur_spk_name] = self.spk_cnt[cur_spk_name]-1
            return {'spk_name':cur_spk_name,'spk_embed':spk_embed,'emotion':emotion}
            

        
        # save_dir = f'./inference/infer_out/{hparams["exp_name"]}/{batch_idx}'
        # os.makedirs(save_dir,exist_ok=True)
        
        # model_out = self.run_model(sample, infer=True)
        # spk_embed = model_out['spk_embed'].detach().cpu().numpy()
        
        # mel = sample['mels'][0].cpu().numpy()
        # spk_mel = sample['ref_spk_mels'][0].cpu().numpy()
        # gen_mel = model_out['mel_out'][0].cpu().numpy()

        # wav = self.vocoder.spec2wav(mel)
        # spk_wav = self.vocoder.spec2wav(spk_mel)
        # gen_wav = self.vocoder.spec2wav(gen_mel)
        
        
        # self.spec_to_figure(gen_mel,vmin=hparams['mel_vmin'],vmax=hparams['mel_vmax'],save_path=f'{save_dir}/gem_mel.png')
        # save_wav(wav,f'{save_dir}/gt.wav',sr=16000)
        # save_wav(spk_wav,f'{save_dir}/ref_spk.wav',sr=16000)
        # save_wav(gen_wav,f'{save_dir}/gen.wav',sr=16000)
        return None
    
    def test_end(self, outputs):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        # 假设你有一个名为 embeddings 的数组，其中包含你的 embedding 数据
        # 假设 labels 是对应的标签
        # 请替换为你自己的实际数据
        labels =[]
        emotion_labels = []
        spk_embed_list = []
        name_id_map = {}
        emotions = ["surprised", "contempt", "sad", "neutral", "disgusted", "happy", "angry", "fear"]
        emotion_id_map = {emo:i  for i,emo in enumerate(emotions)}
        for i,spk_name in enumerate(self.spk_names):
            name_id_map[spk_name]=i
        for out in outputs:
            if out!=None:
                spk_embed_list.append(out['spk_embed'])
                labels.append(name_id_map[out['spk_name']])
                emotion_labels.append(emotion_id_map[out['emotion']])
        nums = len(self.spk_names)
        labels = np.array(labels)
        # nums = len(emotion_id_map)
        # labels = np.array(emotion_labels)

        embeddings = np.concatenate(spk_embed_list,axis=0)

        # 使用 t-SNE 对 embedding 进行降维
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(embeddings)

        # 绘制聚类图
        plt.figure(figsize=(8, 6))

        # 假设你有 10 个类别
        for i in range(nums):
            indices = labels == i
            plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=f'{self.spk_names[i]}')
            # plt.scatter(embedded_data[indices, 0], embedded_data[indices, 1], label=f'{emotions[i]}')

        plt.title('t-SNE Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        save_dir = f'/home/ywx1252669/VC_CLAP/inference/infer_out/{hparams["exp_name"]}'
        os.makedirs(save_dir,exist_ok=True)
        plt.savefig(f'{save_dir}/spk_embedddings_emotion.png')
        plt.show()

        return {}

    def spec_to_figure(self,spec, vmin=None, vmax=None, title='', f0s=None, save_path=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        LINE_COLORS = ['w', 'r', 'orange', 'k', 'cyan', 'm', 'b', 'lime', 'g', 'brown', 'navy']
        
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        H = spec.shape[1] // 2
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        if f0s is not None:
            ax = plt.gca()
            ax2 = ax.twinx()
            if not isinstance(f0s, dict):
                f0s = {'f0': f0s}
            for i, (k, f0) in enumerate(f0s.items()):
                if isinstance(f0, torch.Tensor):
                    f0 = f0.cpu().numpy()
                ax2.plot(f0, label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.5)
            ax2.set_ylim(0, 1000)
            ax2.legend()
        fig.savefig(save_path)

    def _training_step(self, sample, batch_idx, optimizer_idx):       
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output
        
    def plot_wav(self, batch_idx, wav, label, is_mel=False, name=None):
        wav = wav[0].cpu().numpy()
        if is_mel:
            wav_out = self.vocoder.spec2wav(wav)
        self.logger.add_audio(label, wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)