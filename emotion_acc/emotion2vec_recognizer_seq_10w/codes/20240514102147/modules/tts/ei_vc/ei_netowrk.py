from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from utils.commons.hparams import hparams
from modules.commons.conv import TextConvEncoder, ConvBlocks
from modules.commons.layers import Embedding
from modules.commons.rel_transformer import RelTransformerEncoder
from modules.commons.rnn import TacotronEncoder, RNNEncoder, DecoderRNN
from modules.commons.transformer import FastSpeechEncoder, FastSpeechDecoder,ConditionalFastSpeechDecoder
from modules.commons.wavenet import WN
from modules.tts.ei_vc.mel_encoder import EI_Encoder


FS_ENCODERS = {
    'fft': lambda hp, dict_size: FastSpeechEncoder(
        dict_size, hp['hidden_size'], hp['enc_layers'], hp['enc_ffn_kernel_size'],
        num_heads=hp['num_heads']),
    'tacotron': lambda hp, dict_size: TacotronEncoder(
        hp['hidden_size'], dict_size, hp['hidden_size'],
        K=hp['encoder_K'], num_highways=4, dropout=hp['dropout']),
    'tacotron2': lambda hp, dict_size: RNNEncoder(dict_size, hp['hidden_size']),
    'conv': lambda hp, dict_size: TextConvEncoder(dict_size, hp['hidden_size'], hp['hidden_size'],
                                                  hp['enc_dilations'], hp['enc_kernel_size'],
                                                  layers_in_block=hp['layers_in_block'],
                                                  norm_type=hp['enc_dec_norm'],
                                                  post_net_kernel=hp.get('enc_post_net_kernel', 3)),
    'rel_fft': lambda hp, dict_size: RelTransformerEncoder(
        dict_size,hp['code_dim'], hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
}

FS_DECODERS = {
    'fft': lambda hp: FastSpeechDecoder(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
    'rnn': lambda hp: DecoderRNN(hp['hidden_size'], hp['decoder_rnn_dim'], hp['dropout']),
    'cond_fft': lambda hp: ConditionalFastSpeechDecoder(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
    'conv': lambda hp: ConvBlocks(hp['hidden_size'], hp['hidden_size'], hp['dec_dilations'],
                                  hp['dec_kernel_size'], layers_in_block=hp['layers_in_block'],
                                  norm_type=hp['enc_dec_norm'], dropout=hp['dropout'],
                                  post_net_kernel=hp.get('dec_post_net_kernel', 3)),
    'wn': lambda hp: WN(hp['hidden_size'], kernel_size=5, dilation_rate=1, n_layers=hp['dec_layers'],
                        is_BTC=True),
}


class EI_Network(nn.Module):
    def __init__(self, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.hidden_size = hparams['hidden_size']
        # self.codebook = nn.Embedding(hparams['num_codes']+1,hparams['code_dim'],padding_idx=hparams['num_codes'])
        self.content_encoder = FS_ENCODERS[hparams['encoder_type']](hparams, hparams['num_codes'])
        self.ei_encoder = EI_Encoder(mel_dim=hparams['audio_num_mel_bins'],style_dim=hparams['style_dim'],neck_dim=hparams['neck_dim'])
        
        self.timbre_proj = nn.Linear(hparams['style_dim'], hparams['hidden_size'], bias=True)
        self.emo_proj = nn.Linear(hparams['style_dim'], hparams['hidden_size'], bias=True)
        self.code_proj = nn.Linear(hparams['style_dim'], hparams['hidden_size'], bias=True)
        
        if hparams.get('use_adpative_cond_fft',False):
            self.adaptive_encoder = FS_DECODERS[hparams.get('adaptive_encoder_type','cond_fft')](hparams)

        # timbre
        self.timbre_decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.timbre_mel_out = nn.Linear(self.hidden_size, hparams['audio_num_mel_bins'], bias=True)
        # emo
        self.emo_decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.emo_mel_out = nn.Linear(self.hidden_size, hparams['audio_num_mel_bins'], bias=True)
        

    def forward(self, mel,drive_mel,mel_unit,drive_mel_unit,skip_decoder=True, stage='train', **kwargs):
        ret = {}
        # assert mel.shape[1]==mel_unit.shape[1],'error mel length'
        # assert drive_mel.shape[1]==dirve_mel_unit.shape[1],'error mel length'
        mel_mask = (mel.abs().sum(dim=-1)>0).float().unsqueeze(dim=1)  # (b,c,t)
        drive_mel_mask = (drive_mel.abs().sum(dim=-1)>0).float().unsqueeze(dim=1)
        
        h_ling = self.content_encoder(mel_unit,mel_mask)
        h_ling_drive = self.content_encoder(drive_mel_unit,drive_mel_mask)
        
        ei_code,e_timbre,e_emo = self.ei_encoder(mel)
        ret['e_timbre'] = e_timbre
        ret['e_emo'] = e_emo
        e_timbre = self.timbre_proj(e_timbre)
        e_emo = self.emo_proj(e_emo)
        ei_code = self.code_proj(ei_code)
        ei_code_drive,e_timbre_drive,e_emo_drive = self.ei_encoder(drive_mel)
        ret['e_timbre_drive'] = e_timbre_drive
        ret['e_emo_drive'] = e_emo_drive
        e_timbre_drive = self.timbre_proj(e_timbre_drive)
        e_emo_drive = self.emo_proj(e_emo_drive)
        ei_code_drive = self.code_proj(ei_code_drive)

        if stage == 'train':
            
            # forward
            latent_timbreD = h_ling + e_timbre_drive.unsqueeze(dim=1) + ei_code.unsqueeze(dim=1)
            cond = e_timbre_drive + ei_code
            mel_recon = self.forward_timbre_decoder(latent_timbreD, mel_mask,cond)
            ret['fake_timbreB2A'] = mel_recon
            if hparams.get('middle_detach',False):
                ei_code1,e_timbre1,e_emo1 = self.ei_encoder(ret['fake_timbreB2A'].detach())
            else:
                ei_code1,e_timbre1,e_emo1 = self.ei_encoder(ret['fake_timbreB2A'])
            ret['e_timbre1'] = e_timbre1
            ret['e_emo1'] = e_emo1
            e_timbre1,e_emo1,ei_code1 = self.timbre_proj(e_timbre1) , self.emo_proj(e_emo1),self.code_proj(ei_code1)
            latent_timbreemoD = h_ling + ei_code1.unsqueeze(dim=1) + e_emo_drive.unsqueeze(dim=1)
            cond = ei_code1 + e_emo_drive
            mel_recon = self.forward_emo_decoder(latent_timbreemoD, mel_mask,cond)
            ret['fake_timbre_emoB2A'] = mel_recon


            # backward
            latent_emoS = h_ling_drive + ei_code_drive.unsqueeze(dim=1) + e_emo.unsqueeze(dim=1)
            cond = ei_code_drive + e_emo
            mel_recon = self.forward_emo_decoder(latent_emoS, drive_mel_mask,cond)
            ret['fake_emoA2B'] = mel_recon
            if hparams.get('middle_detach',False):
                ei_code2,e_timbre2,e_emo2 = self.ei_encoder(ret['fake_emoA2B'].detach())
            else:
                ei_code2,e_timbre2,e_emo2 = self.ei_encoder(ret['fake_emoA2B'])
            ret['e_timbre2'] = e_timbre2
            ret['e_emo2'] = e_emo2
            e_timbre2,e_emo2,ei_code2 = self.timbre_proj(e_timbre2) , self.emo_proj(e_emo2), self.code_proj(ei_code2)
            latent_emotimbreS = h_ling_drive + e_timbre.unsqueeze(dim=1) + ei_code2.unsqueeze(dim=1)
            cond = e_timbre + ei_code2
            mel_recon = self.forward_timbre_decoder(latent_emotimbreS, drive_mel_mask,cond)
            ret['fake_emo_timbreA2B'] = mel_recon
            
            # dual recon
            latent_dual_forward = h_ling + e_timbre.unsqueeze(dim=1)  + ei_code1.unsqueeze(dim=1)
            cond = e_timbre  + ei_code1
            mel_recon = self.forward_timbre_decoder(latent_dual_forward, mel_mask,cond)
            ret['fake_dual_forward'] = mel_recon
            latent_dual_back = h_ling_drive + ei_code2.unsqueeze(dim=1)  + e_emo_drive.unsqueeze(dim=1)
            cond = ei_code2  + e_emo_drive
            mel_recon = self.forward_emo_decoder(latent_dual_back, drive_mel_mask,cond)
            ret['fake_dual_back'] = mel_recon

            # self rec
            # timbre
            latent_selftimbre = h_ling + ei_code.unsqueeze(dim=1) + e_timbre.unsqueeze(dim=1)
            cond = ei_code + e_timbre
            mel_recon = self.forward_timbre_decoder(latent_selftimbre, mel_mask,cond)
            ret['fake_selftimbre'] = mel_recon

            # emo
            latent_selfemo = h_ling + e_emo.unsqueeze(dim=1) + ei_code.unsqueeze(dim=1)
            cond = e_emo + ei_code
            mel_recon = self.forward_emo_decoder(latent_selfemo, mel_mask,cond)
            ret['fake_selfemo'] = mel_recon

            return ret
        elif stage=='timbre':
            latent_timbreD = h_ling + e_timbre_drive.unsqueeze(dim=1) + ei_code.unsqueeze(dim=1)
            cond = e_timbre_drive + ei_code
            mel_recon = self.forward_timbre_decoder(latent_timbreD, mel_mask,cond)
        elif stage == 'emo':
            latent_emoD = h_ling + e_emo_drive.unsqueeze(dim=1) + ei_code.unsqueeze(dim=1)
            cond = e_emo_drive + ei_code
            mel_recon = self.forward_emo_decoder(latent_emoD, mel_mask,cond)
        elif stage=='both':
            latent_timbreD = h_ling + e_timbre_drive.unsqueeze(dim=1) + ei_code.unsqueeze(dim=1)
            cond = e_timbre_drive + ei_code
            mel_recon = self.forward_timbre_decoder(latent_timbreD, mel_mask,cond)
            ei_code1,e_timbre1,e_emo1 = self.ei_encoder(mel_recon)
            e_timbre1,e_emo1,ei_code1 = self.timbre_proj(e_timbre1) , self.emo_proj(e_emo1),self.code_proj(ei_code1)
            latent_timbreemoD = h_ling + ei_code1.unsqueeze(dim=1) + e_emo_drive.unsqueeze(dim=1)
            cond = ei_code1 + e_emo_drive
            mel_recon = self.forward_emo_decoder(latent_timbreemoD, mel_mask,cond)
        else:
            print("---------------------------ERROR------------------------")
            exit(0)
        return mel_recon

    def forward_style_embed(self, spk_embed=None, spk_id=None):
        # add spk embed
        style_embed = 0
        if self.hparams['use_spk_embed']:
            style_embed = style_embed + self.spk_embed_proj(spk_embed)[:, None, :]
        if self.hparams['use_spk_id']:
            style_embed = style_embed + self.spk_id_proj(spk_id)[:, None, :]
        return style_embed

    def forward_timbre_decoder(self, decoder_inp, tgt_nonpadding,cond, **kwargs):
        if hparams.get('use_adpative_cond_fft',False):
            x = self.adaptive_encoder(decoder_inp,cond.unsqueeze(dim=1))
        else:
            x = decoder_inp  # [B, T, C]
        if hparams.get('use_cln',True):
            x = self.timbre_decoder(x,cond.unsqueeze(dim=1))
        else:
            x = self.timbre_decoder(x)
        x = self.timbre_mel_out(x)
        return x * tgt_nonpadding.transpose(1,2)
    
    def forward_emo_decoder(self, decoder_inp, tgt_nonpadding,cond, **kwargs):
        if hparams.get('use_adpative_cond_fft',False):
            x = self.adaptive_encoder(decoder_inp,cond.unsqueeze(dim=1))
        else:
            x = decoder_inp  # [B, T, C]  
        if hparams.get('use_cln',True):
            x = self.emo_decoder(x,cond.unsqueeze(dim=1))
        else:
            x = self.emo_decoder(x)
        x = self.emo_mel_out(x)
        return x * tgt_nonpadding.transpose(1,2)
