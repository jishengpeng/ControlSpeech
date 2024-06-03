import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils.commons.hparams import hparams

from tasks.tts.fs import FastSpeechTask
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls, BaseVocoder
from modules.tts.ei_vc.disc.emotion_disc import Emotion_Discriminator,Emotion2Vec_finetume
from utils.audio.pitch.utils import denorm_f0
from tasks.tts.vc_editor_dataset_utils import RecognizerDataset,Emotion2VecDataset



class EmotionDiscSpeechTask(FastSpeechTask):
    def __init__(self):
        super(EmotionDiscSpeechTask, self).__init__()
        self.dataset_cls = RecognizerDataset
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        self.criterion = torch.nn.CrossEntropyLoss()


    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = Emotion_Discriminator(hparams['audio_num_mel_bins'],hparams['emo_num'])
        models = [self.model]
        self.model_params = []
        for model in models:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.model_params.append(param)


    def run_model(self, sample, infer=False):
        mel = sample['mels']  # [B, T_s, 80]
        emotion = sample['emotion']
        spk = sample['spk']
        
        if not infer:
            output = self.model(mel)
            losses = {}
            self.add_cls_loss(output['logits'], emotion,losses=losses)
            accuracy = self.compute_accuracy(output['logits'].detach(), emotion)
            output['acc'] =  accuracy
            return losses, output
        else:
            output = self.model(mel)
            return output
   
    def compute_accuracy(self,output,labels):
        output = nn.functional.softmax(output,dim=1) 
        preds = torch.argmax(output,dim=1) # 获取每个样本的预测标签
        correct = torch.sum(preds == labels).item() # 计算正确预测的数量
        accuracy = correct / len(labels) # 除以总样本数得到准确率
        return accuracy
    
    def add_cls_loss(self,pred,tgt_label,losses=None):
        emo_cls_loss = self.criterion(pred,tgt_label)
        losses['emo_cls_loss'] = emo_cls_loss
        
    
    def _training_step(self, sample, batch_idx, optimizer_idx):       
        loss_output, model_out = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        ### 添加单批次的分类准确率
        loss_output['acc'] = model_out['acc']
        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs['losses']['acc'] = model_out['acc']
        outputs = utils.commons.tensor_utils.tensors_to_scalars(outputs)        
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
        return [optimizer_gen]
    
    def build_scheduler(self, optimizer):
        return [
            FastSpeechTask.build_scheduler(self, optimizer[0]), 
        ]
    
    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.model_params, hparams['clip_grad_norm'])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        assert self.scheduler is not None,'scheduler is None,error!!!!'
        if optimizer_idx == 0:
            self.scheduler[0].step(self.global_step// hparams['accumulate_grad_batches'])


class Emotion2VecDiscSpeechTask(FastSpeechTask):
    def __init__(self):
        super(Emotion2VecDiscSpeechTask, self).__init__()
        self.dataset_cls = Emotion2VecDataset
        self.criterion = torch.nn.CrossEntropyLoss()


    def build_model(self):
        self.build_tts_model()
        utils.nn.model_utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = Emotion2Vec_finetume(768,hparams['emo_num'])
        models = [self.model]
        self.model_params = []
        for model in models:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.model_params.append(param)


    def run_model(self, sample, infer=False):
        emo_embeds = sample['emo_embeds']
        emotion = sample['emotion']
        spk = sample['spk']
        
        if not infer:
            output = self.model(emo_embeds)
            losses = {}
            self.add_cls_loss(output['logits'], emotion,losses=losses)
            accuracy = self.compute_accuracy(output['logits'].detach(), emotion)
            output['acc'] =  accuracy
            return losses, output
        else:
            output = self.model(mel)
            return output
   
    def compute_accuracy(self,output,labels):
        output = nn.functional.softmax(output,dim=1) 
        preds = torch.argmax(output,dim=1) # 获取每个样本的预测标签
        correct = torch.sum(preds == labels).item() # 计算正确预测的数量
        accuracy = correct / len(labels) # 除以总样本数得到准确率
        return accuracy
    
    def add_cls_loss(self,pred,tgt_label,losses=None):
        emo_cls_loss = self.criterion(pred,tgt_label)
        losses['emo_cls_loss'] = emo_cls_loss
        
    
    def _training_step(self, sample, batch_idx, optimizer_idx):       
        loss_output, model_out = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        ### 添加单批次的分类准确率
        loss_output['acc'] = model_out['acc']
        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs['losses']['acc'] = model_out['acc']
        outputs = utils.commons.tensor_utils.tensors_to_scalars(outputs)        
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
        return [optimizer_gen]
    
    def build_scheduler(self, optimizer):
        return [
            FastSpeechTask.build_scheduler(self, optimizer[0]), 
        ]
    
    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(self.model_params, hparams['clip_grad_norm'])

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        assert self.scheduler is not None,'scheduler is None,error!!!!'
        if optimizer_idx == 0:
            self.scheduler[0].step(self.global_step// hparams['accumulate_grad_batches'])
