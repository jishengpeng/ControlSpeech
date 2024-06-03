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
from modules.tts.ei_vc.mel_encoder import ECAPA_TDNN_SMALL
from modules.tts.ei_vc.disc.emotion_disc import Emotion_Discriminator
from modules.tts.ei_vc.disc.timbre_disc import Timbre_Discriminator
from utils.audio.pitch.utils import denorm_f0
from tasks.tts.vc_editor_dataset_utils import EIEditorDataset
from modules.tts.ei_vc.mi_estimators import CLUB
from modules.tts.ps_adv.multi_window_disc import Discriminator



class EIEditorSpeechTask(FastSpeechTask):
    def __init__(self):
        super(EIEditorSpeechTask, self).__init__()
        self.dataset_cls = EIEditorDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self.criterion = nn.L1Loss(reduction='mean')
        self.mse_loss_fn = torch.nn.MSELoss()
        self.build_disc_model()


    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model
    
    def build_disc_model(self):
        disc_win_num = hparams['disc_win_num']
        h = hparams['mel_disc_hidden_size']
        self.mel_disc = Discriminator(
            time_lengths=[32, 64, 128][:disc_win_num],
            freq_length=80, hidden_size=h, kernel=(3, 3)
        )
        self.disc_params = list(self.mel_disc.parameters())

    def build_tts_model(self):
        self.model = EI_Network(hparams,out_dims=['audio_num_mel_bins'])
        # self.timbre_disc = Timbre_Discriminator(hparams['audio_num_mel_bins'],hparams['spk_num'])
        # self.emo_disc = Emotion_Discriminator(hparams['audio_num_mel_bins'],hparams['emo_num'])  
        self.timbre_disc = ECAPA_TDNN_SMALL(hparams['audio_num_mel_bins'])
        self.emo_disc = ECAPA_TDNN_SMALL(hparams['audio_num_mel_bins'])
        # self.ti_emo_mi_net = CLUB(hparams['style_dim'],hparams['style_dim'],hparams['mi_dim'])
        models = [self.model,self.emo_disc,self.timbre_disc]
        self.model_params = []
        # self.mi_params = list(self.ti_emo_mi_net.parameters())
        # self.load_discriminators(self.emo_disc,'1115_emotion_recognizer')
        # self.load_discriminators(self.timbre_disc,'1115_timbre_recognizer')
        for model in models:
            for name, param in model.named_parameters():
                self.model_params.append(param)
                    
    def load_discriminators(self,model,ckpt_name):
        from utils.commons.ckpt_utils import get_last_checkpoint, get_all_ckpts
        checkpoint, _ = get_last_checkpoint(f'checkpoints/{ckpt_name}', None)
        if checkpoint is not None:
            # 加载预训练模型timbre encoder/emotion encoder
            model_dict=model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict']['model'].items()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        #  frozen params
        for k,v in model.named_parameters():
            v.requires_grad=False

    def run_model(self, sample, infer=False,stage='train'):
        mel = sample['mels']  # [B, T_s, 80]
        drive_mel = sample['drive_ref_mels']
        mel_unit = sample['unit_ids']
        drive_mel_unit = sample['drive_ref_units']
        spk_embed = sample.get('spk_embed') if hparams['use_spk_embed'] else None
        spk_id = sample.get('spk_ids') if hparams['use_spk_id'] else None
        
        if not infer:
            output = self.model(mel,drive_mel,mel_unit,drive_mel_unit,stage=stage)
            losses = {}
            self.add_reconstruction_loss(sample,output,losses)
            self.add_pair_loss(sample,output,losses)
            # self.add_orthogonality_loss(output,losses)
            # self.add_one_emo_loss(sample,output,losses)
            # self.add_one_timbre_loss(sample,output,losses)
            # self.add_mel_loss(output['mel_out'], target, losses)
            # self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
            return losses, output
        else:
            output = self.model(mel,drive_mel,mel_unit,drive_mel_unit,stage=stage)
            return output
    
    def add_orthogonality_loss(self,model_out,losses):
        e_timbre,e_emo = model_out['e_timbre'],model_out['e_emo']
        e_timbre_drive,e_emo_drive = model_out['e_timbre_drive'],model_out['e_emo_drive']
        e_timbre1,e_emo1 = model_out['e_timbre1'],model_out['e_emo1'] 
        e_timbre2,e_emo2 = model_out['e_timbre2'], model_out['e_emo2']
        l1 = torch.norm(torch.matmul(e_timbre.unsqueeze(dim=1),e_emo.unsqueeze(dim=-1)),p=2)
        l2 = torch.norm(torch.matmul(e_timbre_drive.unsqueeze(dim=1),e_emo_drive.unsqueeze(dim=-1)),p=2)
        l3 = torch.norm(torch.matmul(e_timbre1.unsqueeze(1),e_emo1.unsqueeze(-1)),p=2)
        l4 = torch.norm(torch.matmul(e_timbre2.unsqueeze(1),e_emo2.unsqueeze(-1)),p=2)
        losses['ortho_loss'] = (l1+l2+l3+l4)*hparams.get('lambda_orthogonality_loss',1.0)
    
    def cycle_consistence(self,mel1,mel2):
        p1_code,p1_timbre,p1_emo = self.model.ei_encoder(mel1)
        p2_code,p2_timbre,p2_emo = self.model.ei_encoder(mel2)
        timbre_clc_loss = F.mse_loss(p1_timbre,p2_timbre)
        emo_clc_loss = F.mse_loss(p1_emo,p2_emo)
        code_clc_loss = self.criterion(p1_code,p2_code)
        return timbre_clc_loss,emo_clc_loss,code_clc_loss
    
    def cal_disc_loss(self,mel1,mel2):
        mel1 = mel1.transpose(1,2)
        mel2 = mel2.transpose(1,2)
        p1_timbre,p1_emo = self.timbre_disc(mel1),self.emo_disc(mel1)
        p2_timbre,p2_emo = self.timbre_disc(mel2),self.emo_disc(mel2)
        timbre_disc_loss = self.criterion(p1_timbre,p2_timbre)
        emo_disc_loss = self.criterion(p1_emo,p2_emo)
        return timbre_disc_loss,emo_disc_loss
    
    def add_one_emo_loss(self,sample,model_out,losses):
        mel1 = sample['mels'].transpose(1,2)
        mel2 = model_out['fake_timbreB2A'].transpose(1,2)
        p1_emo = self.emo_disc(mel1)
        p2_emo = self.emo_disc(mel2)
        emo_disc_loss = self.criterion(p1_emo,p2_emo)
        losses['one_emo_loss'] = emo_disc_loss * hparams.get('lambda_one_emo_loss',1.0)
    
    def add_one_timbre_loss(self,sample,model_out,losses):
        mel1 = sample['drive_ref_mels'].transpose(1,2)
        mel2 = model_out['fake_emoA2B'].transpose(1,2)
        p1_timbre = self.timbre_disc(mel1)
        p2_timbre = self.timbre_disc(mel2)
        timbre_disc_loss = self.criterion(p1_timbre,p2_timbre)
        losses['one_timbre_loss'] = timbre_disc_loss * hparams.get('lambda_one_timbre_loss',1.0)        
        
    # def cal_disc_loss(self,mel1,mel2):
    #     p1_timbre,p1_emo = self.timbre_disc(mel1)['timbre_embed'],self.emo_disc(mel1)['emo_embed']
    #     p2_timbre,p2_emo = self.timbre_disc(mel2)['timbre_embed'],self.emo_disc(mel2)['emo_embed']
    #     p1_timbre,p1_emo = self.timbre_disc(mel1)['timbre_embed'],self.emo_disc(mel1)['emo_embed']
    #     p2_timbre,p2_emo = self.timbre_disc(mel2)['timbre_embed'],self.emo_disc(mel2)['emo_embed']
    #     timbre_disc_loss = self.criterion(p1_timbre,p2_timbre)
    #     emo_disc_loss = self.criterion(p1_emo,p2_emo)
    #     return timbre_disc_loss,emo_disc_loss
    
    def add_reconstruction_loss(self,sample,model_out,losses):
        self.add_mel_loss(model_out['fake_dual_forward'],sample['mels'],losses,postfix='dualA')
        self.add_mel_loss(model_out['fake_dual_back'],sample['drive_ref_mels'],losses,postfix='dualB')
        self.add_mel_loss(model_out['fake_selftimbre'],sample['mels'],losses,postfix='selftimbre')
        self.add_mel_loss(model_out['fake_selfemo'],sample['mels'],losses,postfix='selfemo')
    
    def add_pair_loss(self,sample,model_out,losses):
        timbre_disc_loss1,emo_disc_loss1 = self.cal_disc_loss(model_out['fake_timbreB2A'],model_out['fake_emoA2B'])
        timbre_disc_loss2,emo_disc_loss2 = self.cal_disc_loss(model_out['fake_timbre_emoB2A'],sample['drive_ref_mels'])
        timbre_disc_loss3,emo_disc_loss3 = self.cal_disc_loss(model_out['fake_emo_timbreA2B'],sample['mels'])
        timbre_disc_loss = timbre_disc_loss1 + timbre_disc_loss2 + timbre_disc_loss3
        emo_disc_loss = emo_disc_loss1 + emo_disc_loss2 + emo_disc_loss3
        losses['pair_loss'] = (timbre_disc_loss + emo_disc_loss)* hparams['lambda_pair_loss']
    
    
    # def add_pair_loss(self,sample,model_out,losses):
    #     code_clc_loss1,timbre_clc_loss1,emo_clc_loss1 = self.cycle_consistence(model_out['fake_timbreB2A'],model_out['fake_emoA2B'])
    #     code_clc_loss2,timbre_clc_loss2,emo_clc_loss2 = self.cycle_consistence(model_out['fake_timbre_emoB2A'],sample['drive_ref_mels'])
    #     code_clc_loss3,timbre_clc_loss3,emo_clc_loss3 = self.cycle_consistence(model_out['fake_emo_timbreA2B'],sample['mels'])
    #     timbre_clc_loss = timbre_clc_loss1 + timbre_clc_loss2 + timbre_clc_loss3
    #     emo_clc_loss = emo_clc_loss1 + emo_clc_loss2 + emo_clc_loss3
    #     code_clc_loss = code_clc_loss1 + code_clc_loss2 + code_clc_loss3
    #     losses['pair_loss'] = (timbre_clc_loss + emo_clc_loss + code_clc_loss)* hparams['lambda_pair_loss']
        

    def validation_step(self, sample, batch_idx):
        outputs = {}
        mel = sample['mels']  # [B, T_s, 80]
        drive_mel = sample['drive_ref_mels']
        mel_unit = sample['unit_ids']
        drive_mel_unit = sample['drive_ref_units']

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.commons.tensor_utils.tensors_to_scalars(outputs)
        ## 音色转换、情绪转换、两者都转换
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(mel,drive_mel,mel_unit,drive_mel_unit)
                
            self.plot_mel(batch_idx, sample['mels'], model_out['fake_dual_forward'],name=f'mel_val_{batch_idx}_dualA')
            self.plot_mel(batch_idx, sample['drive_ref_mels'], model_out['fake_dual_back'],name=f'mel_val_{batch_idx}_dualB')
            self.plot_mel(batch_idx, sample['mels'], model_out['fake_selftimbre'],name=f'mel_val_{batch_idx}_selftimbre')
            self.plot_mel(batch_idx, sample['mels'], model_out['fake_selfemo'],name=f'mel_val_{batch_idx}_selfemo')
            self.plot_mel(batch_idx, model_out['fake_timbreB2A'], model_out['fake_emoA2B'],name=f'mel_val_{batch_idx}_midpair')
            self.plot_mel(batch_idx, model_out['fake_timbre_emoB2A'], model_out['fake_emo_timbreA2B'],name=f'mel_val_{batch_idx}_endpair')
            ## source wav, drive wav
            self.plot_wav(batch_idx, sample['mels'], sample['drive_ref_mels'],f'gt_{batch_idx}_source',f'wav_{batch_idx}_drive', is_mel=True)
            self.plot_wav(batch_idx, model_out['fake_timbreB2A'],model_out['fake_emoA2B'],f'wav_{batch_idx}_timbreB2A',f'wav_{batch_idx}_emoA2B', is_mel=True)
            self.plot_wav(batch_idx, model_out['fake_timbre_emoB2A'],model_out['fake_emo_timbreA2B'],f'wav_{batch_idx}_timbre_emoB2A',f'wav_{batch_idx}_emo_timbreA2B', is_mel=True)
            self.plot_wav(batch_idx, model_out['fake_dual_forward'],model_out['fake_dual_back'],f'wav_{batch_idx}_dualA',f'wav_{batch_idx}_dualB', is_mel=True)
            self.plot_wav(batch_idx, model_out['fake_selftimbre'],model_out['fake_selfemo'],f'wav_{batch_idx}_selftimbre',f'wav_{batch_idx}_selfemo', is_mel=True)
            
        return outputs
    
    def test_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return:
        """
        assert sample['mels'].shape[0] == 1, 'only support batch_size=1 in inference'
        stage_map = {0:'timbre',1:'emo',2:'both'}
        import random
        from utils.audio.io import save_wav
        random_number = random.randint(0, 2)
        stage = stage_map[random_number]
        save_dir = f'./inference/infer_out/{hparams["exp_name"]}/{stage}/{batch_idx}'
        os.makedirs(save_dir,exist_ok=True)
        
        mel_recon = self.run_model(sample, infer=True,stage=stage)
        
        mel = sample['mels'][0].cpu().numpy()
        drive_mel = sample['drive_ref_mels'][0].cpu().numpy()
        gen_mel = mel_recon[0].cpu().numpy()

        source_wav = self.vocoder.spec2wav(mel)
        drive_wav = self.vocoder.spec2wav(drive_mel)
        gen_wav = self.vocoder.spec2wav(gen_mel)
        save_wav(source_wav,f'{save_dir}/source.wav',sr=16000)
        save_wav(drive_wav,f'{save_dir}/drive.wav',sr=16000)
        save_wav(gen_wav,f'{save_dir}/gen.wav',sr=16000)
        
        return {}
    
    def build_optimizer(self, model):
        
        optimizer_gen = torch.optim.AdamW(
            self.model_params,
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])

        optimizer_disc = torch.optim.AdamW(
            self.disc_params,
            lr=hparams['disc_lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            **hparams["discriminator_optimizer_params"]) if len(self.disc_params) > 0 else None
        return [optimizer_gen,optimizer_disc]
    
    def build_scheduler(self, optimizer):
        return [
            FastSpeechTask.build_scheduler(self, optimizer[0]), # Generator Scheduler
            torch.optim.lr_scheduler.StepLR(optimizer=optimizer[1], # Discriminator Scheduler
                **hparams["discriminator_scheduler_params"]),
            # FastSpeechTask.build_scheduler(self, optimizer[1]),
        ]
    
    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.model_params, hparams['clip_grad_norm'])
        else:
            nn.utils.clip_grad_norm_(self.disc_params, hparams["clip_grad_norm"])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        assert self.scheduler is not None,'scheduler is None,error!!!!'
        if optimizer_idx == 0:
            self.scheduler[0].step(self.global_step// hparams['accumulate_grad_batches'])
        else:
            self.scheduler[1].step(self.global_step // hparams['accumulate_grad_batches'])
    
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
            loss_output, model_out = self.run_model(sample)
            self.model_out_gt = self.model_out = \
                {k: v.detach() for k, v in model_out.items() if isinstance(v, torch.Tensor)}
            if disc_start:
                self.add_generator_part_loss(model_out['fake_dual_forward'],loss_output,loss_weights,label=1,)
                self.add_generator_part_loss(model_out['fake_dual_back'],loss_output,loss_weights,label=2)
                self.add_generator_part_loss(model_out['fake_selftimbre'],loss_output,loss_weights,label=3)
                self.add_generator_part_loss(model_out['fake_selfemo'],loss_output,loss_weights,label=4)
        else:
            #######################
            #    Discriminator    #
            #######################
            if disc_start and self.global_step % hparams['disc_interval'] == 0:
                self.add_disc_part_loss(sample['mels'],self.model_out_gt['fake_dual_forward'],loss_output,loss_weights,label=1)
                self.add_disc_part_loss(sample['mels'],self.model_out_gt['fake_selftimbre'],loss_output,loss_weights,label=2)
                self.add_disc_part_loss(sample['mels'],self.model_out_gt['fake_selfemo'],loss_output,loss_weights,label=3)
                self.add_disc_part_loss(sample['drive_ref_mels'],self.model_out_gt['fake_dual_back'],loss_output,loss_weights,label=4)
            if len(loss_output) == 0:
                return None
        total_loss = sum([loss_weights.get(k, 1) * v for k, v in loss_output.items() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output
    
    def add_generator_part_loss(self,pred_mel,loss_output,loss_weights,label):
        # mel_p = model_out['mel_out']
        mel_p = pred_mel
        if hasattr(self.model, 'out2mel'):
            mel_p = self.model.out2mel(mel_p)
        o_ = self.mel_disc(mel_p)
        p_, pc_ = o_['y'], o_['y_c']
        if p_ is not None:
            loss_output[f'a_{label}'] = self.mse_loss_fn(p_, p_.new_ones(p_.size()))
            loss_weights[f'a_{label}'] = hparams['lambda_mel_adv']
        if pc_ is not None:
            loss_output[f'ac_{label}'] = self.mse_loss_fn(pc_, pc_.new_ones(pc_.size()))
            loss_weights[f'ac_{label}'] = hparams['lambda_mel_adv']
    
    def add_disc_part_loss(self,mel_g,mel_p,loss_output,loss_weights,label):
        # model_out = self.model_out_gt
        # mel_g = sample['mels']
        # mel_p = model_out['mel_out']
        o = self.mel_disc(mel_g)
        p, pc = o['y'], o['y_c']
        o_ = self.mel_disc(mel_p)
        p_, pc_ = o_['y'], o_['y_c']
        if p_ is not None:
            loss_output[f"r_{label}"] = self.mse_loss_fn(p, p.new_ones(p.size()))
            loss_output[f"f_{label}"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
        if pc_ is not None:
            loss_output[f"rc_{label}"] = self.mse_loss_fn(pc, pc.new_ones(pc.size()))
            loss_output[f"fc_{label}"] = self.mse_loss_fn(pc_, pc_.new_zeros(pc_.size()))
    
    # def on_train_end(self):
    #     if hparams['start_update_recognizer_steps']==self.global_step:
    #         for k,v in self.timbre_disc.named_parameters():
    #             v.requires_grad=True
    #         for k,v in self.emo_disc.named_parameters():
    #             v.requires_grad=True

    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_wav, wav_out, gt_label,pred_label, is_mel=False, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav)
            wav_out = self.vocoder.spec2wav(wav_out)
        self.logger.add_audio(gt_label, gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        self.logger.add_audio(pred_label, wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)