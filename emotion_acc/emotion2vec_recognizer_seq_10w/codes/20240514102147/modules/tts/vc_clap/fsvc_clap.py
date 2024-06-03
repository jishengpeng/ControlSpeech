from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from utils.commons.hparams import hparams
from modules.commons.conv import TextConvEncoder, ConvBlocks
from modules.commons.layers import Embedding
from modules.commons.rel_transformer import RelTransformerEncoder,ConditionalRelTransformerEncoder
from modules.commons.rnn import TacotronEncoder, RNNEncoder, DecoderRNN
from modules.commons.transformer import FastSpeechEncoder, FastSpeechDecoder,ConditionalFastSpeechDecoder,MultiheadAttention
from modules.commons.wavenet import WN
from modules.commons.nar_tts_modules import PitchPredictor, DurationPredictor, LengthRegulator,CondDurationPredictor,CondPitchPredictor
from modules.tts.vc_clap.speaker_encoder import ECAPA_TDNN_SMALL,EqualLinear
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from modules.tts.vc_clap.net import SpeakerClassifier,GradientReversal
from modules.tts.vc_clap.stylebank import StyleBankExtractor
from modules.tts.vc_clap.naturalspeech2 import CondRelTransformerEncoder,DurationPitchPredictor,\
                    TextConditionalRelTransformerEncoder,ConditionableTransformer,WavLMSpeakerAdapter,\
                        MultiStream_Spk_Encoder
from modules.tts.vc_clap.mixstyle import MixStyle


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
        dict_size, hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
    'cond_rel_fft': lambda hp, dict_size: ConditionalRelTransformerEncoder(
        dict_size,hp['code_dim'], hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
    'cond_reltrans': lambda hp, dict_size: CondRelTransformerEncoder(
        dict_size,hp['code_dim'], hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
    'text_cond_reltrans': lambda hp, dict_size: TextConditionalRelTransformerEncoder(
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


class VC_Network(nn.Module):
    def __init__(self, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.hidden_size = hparams['hidden_size']
        # self.codebook = nn.Embedding(hparams['num_codes']+1,hparams['code_dim'],padding_idx=hparams['num_codes'])
        self.content_encoder = FS_ENCODERS[hparams['encoder_type']](hparams, hparams['num_codes'])
        ### speaker encoder
        self.speaker_enc = ECAPA_TDNN_SMALL(hparams.get('spk_mel_dim',20),hparams['hidden_size'])
        # self.spk_in = nn.Linear(hparams['hidden_size']*2, hparams['spk_bsvt_dim'],bias=False)
        self.spk_in = nn.Linear(hparams['hidden_size'], hparams['spk_bsvt_dim'],bias=False)
        self.spk_cross_attention = MultiheadAttention(embed_dim=hparams['spk_bsvt_dim'], num_heads=2)
        self.spk_bank = torch.nn.Parameter(torch.rand((hparams['spk_bsvt_num'], hparams['spk_bsvt_dim'])))
        self.spk_out = nn.Linear(hparams['spk_bsvt_dim'], hparams['hidden_size'],bias=False)
        
        ### style encoder
        bert_dim = hparams['bert_dim']
        fc = [EqualLinear(bert_dim, bert_dim)]
        for i in range(2):
            fc.append(EqualLinear(bert_dim, bert_dim))
        fc.append(EqualLinear(bert_dim, hparams['hidden_size']))
        self.mlp = nn.Sequential(*fc)
        self.style_bank = torch.nn.Parameter(torch.rand((hparams['style_bsvt_num'], hparams['hidden_size'])))
        self.style_cross_attention = MultiheadAttention(embed_dim=hparams['hidden_size'], num_heads=2)
        # self.speaker_classifier = nn.Sequential(
        #     SpeakerClassifier(hparams['hidden_size'],output_dim=1220),
        #     GradientReversal(alpha=1.0)
        #     )
        
        if hparams.get('use_adpative_cond_fft',False):
            self.adaptive_encoder = FS_DECODERS[hparams.get('adaptive_encoder_type','cond_fft')](hparams)
        
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        if hparams['use_cond_predictor']:
            self.dur_predictor = CondDurationPredictor(
                self.hidden_size,
                n_chans=predictor_hidden,
                n_layers=hparams['dur_predictor_layers'],
                dropout_rate=hparams['predictor_dropout'],
                kernel_size=hparams['dur_predictor_kernel'])
        else:
            self.dur_predictor = DurationPredictor(
                self.hidden_size,
                n_chans=predictor_hidden,
                n_layers=hparams['dur_predictor_layers'],
                dropout_rate=hparams['predictor_dropout'],
                kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            if hparams['use_cond_predictor']:
                self.pitch_predictor = CondPitchPredictor(
                    self.hidden_size, n_chans=predictor_hidden,
                    n_layers=5, dropout_rate=0.3, odim=2,
                    kernel_size=hparams['predictor_kernel'])
            else:
                self.pitch_predictor = PitchPredictor(
                    self.hidden_size, n_chans=predictor_hidden,
                    n_layers=5, dropout_rate=0.3, odim=2,
                    kernel_size=hparams['predictor_kernel'])

    def forward(self, mel,ref_mel,ph_token,style_embed,mel2ph,f0,uv,**kwargs):
        ret = {}
        # assert mel.shape[1]==mel_unit.shape[1],'error mel length'
        # assert drive_mel.shape[1]==dirve_mel_unit.shape[1],'error mel length'
        style_embed = self.forward_style_embed(style_embed,ret)
        spk_embed = self.forward_spk_embed(ref_mel,style_embed.squeeze(dim=1),ret)
        cond = style_embed + spk_embed
        
        src_nonpadding = (ph_token<100).float()[:, :,None]    # [B,T,C]
        h_ling = self.content_encoder(ph_token,cond,src_nonpadding.transpose(1,2)) * src_nonpadding
        
        
        # add dur
        dur_inp = h_ling
        mel2ph = self.forward_dur(dur_inp,cond, mel2ph,ph_token, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(h_ling, mel2ph) * tgt_nonpadding

        
        # add pitch embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = decoder_inp 
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,cond, f0, uv, mel2ph, ret, h_ling)
            
        
        if hparams.get('use_adpative_cond_fft',False):
            decoder_inp = self.adaptive_encoder(decoder_inp,spk_embed)
            
        # decoder input
        ret['decoder_inp'] = decoder_inp = (decoder_inp + cond) * tgt_nonpadding
        return ret
    
    def forward_spk_embed(self,ref_mel,style_embed,ret):
        if self.hparams.get('spk_mel_dim',20)==20:
            spk_embed = self.speaker_enc(ref_mel[:,:,:20]) # [B,C]
        else:
            spk_embed = self.speaker_enc(ref_mel)
        # spk_embed = self.spk_in(torch.cat((style_embed,spk_embed),dim=-1)).unsqueeze(dim=1)
        spk_embed = self.spk_in(spk_embed).unsqueeze(dim=1)
        spk_embed = spk_embed.transpose(0,1)   #[1, B, C]
        spk_bank = self.spk_bank
        spk_bank = spk_bank.repeat(spk_embed.shape[1],1,1).transpose(0,1)    #[N, C]  --> [N, B, C]
        spk_embed, attn_weight = self.spk_cross_attention(spk_embed, spk_bank, spk_bank)
        # ret['spk_attn_weight'] = attn_weight
        spk_embed = spk_embed.transpose(0,1)    #[B, 1, C]
        spk_embed = self.spk_out(spk_embed)
        ret['spk_embed'] = spk_embed.unsqueeze(dim=1)
        return spk_embed
        
    def forward_style_embed(self, style_embed,ret):
        style_embed = self.mlp(style_embed).unsqueeze(dim=1)
        style_embed = style_embed.transpose(0,1)   #[1, B, C]
        style_bank = self.style_bank
        style_bank = style_bank.repeat(style_embed.shape[1],1,1).transpose(0,1)    #[N, C]  --> [N, B, C]
        style_embed, attn_weight = self.style_cross_attention(style_embed, style_bank, style_bank)
        ret['style_attn_weight'] = attn_weight
        style_embed = style_embed.transpose(0,1)    #[B, 1, C]
        # predict_out = self.speaker_classifier(style_embed.squeeze(dim=1))
        # ret['predict_out'] = predict_out
        return style_embed
    
    def forward_dur(self, dur_input,cond, mel2ph, ph_token, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param ph_token: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = ph_token == 100
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        if cond is None:
            dur = self.dur_predictor(dur_input, src_padding)
        else: 
            dur = self.dur_predictor(dur_input,cond, src_padding)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph
    
    def forward_pitch(self, decoder_inp,cond, f0, uv, mel2ph, ret, encoder_out=None):
        if self.hparams['pitch_type'] == 'frame':
            pitch_pred_inp = decoder_inp
            pitch_padding = mel2ph == 0
        else:
            pitch_pred_inp = encoder_out
            pitch_padding = encoder_out.abs().sum(-1) == 0
            uv = None
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        if cond is None:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        else:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp,cond)
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            pitch_padding=pitch_padding)
        if self.hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed
    

class VC_Seq_Network(nn.Module):
    def __init__(self, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.hidden_size = hparams['hidden_size']
        # self.codebook = nn.Embedding(hparams['num_codes']+1,hparams['code_dim'],padding_idx=hparams['num_codes'])
        self.content_encoder = FS_ENCODERS[hparams['encoder_type']](hparams, hparams['num_codes'])
        ### speaker encoder
        self.speaker_enc = ECAPA_TDNN_SMALL(hparams.get('spk_mel_dim',20),hparams['hidden_size'])
        # self.spk_in = nn.Linear(hparams['hidden_size']*2, hparams['spk_bsvt_dim'],bias=False)
        self.spk_in = nn.Linear(hparams['hidden_size'], hparams['spk_bsvt_dim'],bias=False)
        self.spk_cross_attention = MultiheadAttention(embed_dim=hparams['spk_bsvt_dim'], num_heads=2)
        self.spk_bank = torch.nn.Parameter(torch.rand((hparams['spk_bsvt_num'], hparams['spk_bsvt_dim'])))
        self.spk_out = nn.Linear(hparams['spk_bsvt_dim'], hparams['hidden_size'],bias=False)
        
        ### style encoder
        bert_dim = hparams['bert_dim']
        fc = [EqualLinear(bert_dim, bert_dim)]
        for i in range(2):
            fc.append(EqualLinear(bert_dim, bert_dim))
        fc.append(EqualLinear(bert_dim, hparams['hidden_size']))
        self.mlp = nn.Sequential(*fc)
        self.style_bank = StyleBankExtractor(hparams['hidden_size'],hparams['sbank_layers'],hparams['sbank_size'],
                                               hparams['sbank_dim'],hparams['sbank_num_heads'])
        
        
        if hparams.get('use_adpative_cond_fft',False):
            self.adaptive_encoder = FS_DECODERS[hparams.get('adaptive_encoder_type','cond_fft')](hparams)
        
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        if hparams['use_cond_predictor']:
            self.dur_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=1)
        else:
            self.dur_predictor = DurationPredictor(
                self.hidden_size,
                n_chans=predictor_hidden,
                n_layers=hparams['dur_predictor_layers'],
                dropout_rate=hparams['predictor_dropout'],
                kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            if hparams['use_cond_predictor']:
                self.pitch_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=2)
            else:
                self.pitch_predictor = PitchPredictor(
                    self.hidden_size, n_chans=predictor_hidden,
                    n_layers=5, dropout_rate=0.3, odim=2,
                    kernel_size=hparams['predictor_kernel'])

    def forward(self, mel,ref_mel,ph_token,style_embed,mel2ph,f0,uv,**kwargs):
        ret = {}
        # assert mel.shape[1]==mel_unit.shape[1],'error mel length'
        # assert drive_mel.shape[1]==dirve_mel_unit.shape[1],'error mel length'
        style_embed = self.forward_style_embed(style_embed,ret)
        spk_embed = self.forward_spk_embed(ref_mel,style_embed,ret)
        cond = style_embed + spk_embed  # [B,32,C]
        
        src_nonpadding = (ph_token<100).float()[:, :,None]    # [B,T,C]
        h_ling = self.content_encoder(ph_token,cond,src_nonpadding.transpose(1,2))
        
        
        # add dur
        dur_inp = h_ling
        mel2ph = self.forward_dur(dur_inp,cond, mel2ph,ph_token, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(h_ling, mel2ph) * tgt_nonpadding

        
        # add pitch embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = decoder_inp 
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,cond, f0, uv, mel2ph, ret, h_ling)
            
        
        if hparams.get('use_adpative_cond_fft',False):
            decoder_inp = self.adaptive_encoder(decoder_inp,spk_embed)
            
        # decoder input
        ### 暂时diff decoder只用spk_embed做cond
        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding
        return ret
    
    def forward_spk_embed(self,ref_mel,style_embed,ret):
        if self.hparams.get('spk_mel_dim',20)==20:
            spk_embed = self.speaker_enc(ref_mel[:,:,:20]) # [B,C]
        else:
            spk_embed = self.speaker_enc(ref_mel)
        # spk_embed = self.spk_in(torch.cat((style_embed,spk_embed),dim=-1)).unsqueeze(dim=1)
        spk_embed = self.spk_in(spk_embed).unsqueeze(dim=1)
        spk_embed = spk_embed.transpose(0,1)   #[1, B, C]
        spk_bank = self.spk_bank
        spk_bank = spk_bank.repeat(spk_embed.shape[1],1,1).transpose(0,1)    #[N, C]  --> [N, B, C]
        spk_embed, attn_weight = self.spk_cross_attention(spk_embed, spk_bank, spk_bank)
        # ret['spk_attn_weight'] = attn_weight
        spk_embed = spk_embed.transpose(0,1)    #[B, 1, C]
        spk_embed = self.spk_out(spk_embed)
        ret['spk_embed'] = spk_embed.unsqueeze(dim=1)
        return spk_embed
        
    def forward_style_embed(self, style_embed,ret):
        style_embed = self.mlp(style_embed)  # [B,T,C]
        padding_mask = (style_embed.sum(dim=-1)).eq(0).data  # [B,T]
        style_embed = style_embed.transpose(0,1)   # [T,B,C]
        style_embed, attn_weights = self.style_bank(style_embed,padding_mask=padding_mask)
        ret['attn_weights'] = attn_weights
        style_embed = style_embed.transpose(0,1)  # [B,T,C]
        return style_embed
    
    def forward_dur(self, dur_input,cond, mel2ph, ph_token, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param ph_token: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = ph_token == 100
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        if cond is None:
            dur = self.dur_predictor(dur_input, src_padding)
        else: 
            dur = self.dur_predictor(dur_input,cond)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph
    
    def forward_pitch(self, decoder_inp,cond, f0, uv, mel2ph, ret, encoder_out=None):
        if self.hparams['pitch_type'] == 'frame':
            pitch_pred_inp = decoder_inp
            pitch_padding = mel2ph == 0
        else:
            pitch_pred_inp = encoder_out
            pitch_padding = encoder_out.abs().sum(-1) == 0
            uv = None
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        if cond is None:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        else:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp,cond)
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            pitch_padding=pitch_padding)
        if self.hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed
        
class TTS_Seq_Network(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.hidden_size = hparams['hidden_size']
        # self.codebook = nn.Embedding(hparams['num_codes']+1,hparams['code_dim'],padding_idx=hparams['num_codes'])
        self.content_encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)
        ### speaker encoder
        self.speaker_enc = ECAPA_TDNN_SMALL(hparams.get('spk_mel_dim',20),hparams['hidden_size'])
        # self.spk_in = nn.Linear(hparams['hidden_size']*2, hparams['spk_bsvt_dim'],bias=False)
        self.spk_in = nn.Linear(hparams['hidden_size'], hparams['spk_bsvt_dim'],bias=False)
        self.spk_cross_attention = MultiheadAttention(embed_dim=hparams['spk_bsvt_dim'], num_heads=2)
        self.spk_bank = torch.nn.Parameter(torch.rand((hparams['spk_bsvt_num'], hparams['spk_bsvt_dim'])))
        print(hparams['spk_bsvt_num'],'*'*50)
        self.spk_out = nn.Linear(hparams['spk_bsvt_dim'], hparams['hidden_size'],bias=False)
        
        ### style encoder
        bert_dim = hparams['bert_dim']
        fc = [EqualLinear(bert_dim, bert_dim)]
        for i in range(2):
            fc.append(EqualLinear(bert_dim, bert_dim))
        fc.append(EqualLinear(bert_dim, hparams['hidden_size']))
        # fc.append(EqualLinear(bert_dim, bert_dim))
        self.mlp = nn.Sequential(*fc)
        # self.style_in = nn.Linear(hparams['bert_dim'],hparams['hidden_size'],bias=False)
        self.style_bank = StyleBankExtractor(hparams['hidden_size'],hparams['sbank_layers'],hparams['sbank_size'],
                                               hparams['sbank_dim'],hparams['sbank_num_heads'])
        
        if hparams.get('use_style_adapter',False):
            print('*'*90)
            self.pre_style_adpater = ConditionableTransformer(
                dim = self.hidden_size,
                depth = 2,
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                ff_causal_conv = True,
                dim_cond_mult = None,
                use_flash = False,
                cross_attn = True
                )
            self.post_style_adapter = ConditionableTransformer(
                dim = self.hidden_size,
                depth = 2,
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                ff_causal_conv = True,
                dim_cond_mult = None,
                use_flash = False,
                cross_attn = True
                )


        if hparams.get('use_adpative_cond_fft',False):
            self.adaptive_encoder = ConditionableTransformer(
                dim = self.hidden_size,
                depth = 2,
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                ff_causal_conv = True,
                dim_cond_mult = None,
                use_flash = False,
                cross_attn = True
                )
        
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        if hparams['use_cond_predictor']:
            self.dur_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=1)
        else:
            self.dur_predictor = DurationPredictor(
                self.hidden_size,
                n_chans=predictor_hidden,
                n_layers=hparams['dur_predictor_layers'],
                dropout_rate=hparams['predictor_dropout'],
                kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            if hparams['use_cond_predictor']:
                self.pitch_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=2)
            else:
                self.pitch_predictor = PitchPredictor(
                    self.hidden_size, n_chans=predictor_hidden,
                    n_layers=5, dropout_rate=0.3, odim=2,
                    kernel_size=hparams['predictor_kernel'])

    def forward(self, mel,ref_mel,ph_token,style_embed,mel2ph,f0,uv,**kwargs):
        ret = {}
        # assert mel.shape[1]==mel_unit.shape[1],'error mel length'
        # assert drive_mel.shape[1]==dirve_mel_unit.shape[1],'error mel length'
        style_embed = self.forward_style_embed(style_embed,ret)
        spk_embed = self.forward_spk_embed(ref_mel,style_embed,ret)
        if self.hparams.get('add_cond',True):
            cond = style_embed + spk_embed  # [B,32,C]
            # cond = style_embed
        else:
            cond = torch.cat((style_embed,spk_embed),dim=1)
        
        src_nonpadding = (ph_token > 0).float()[:, :, None]   # [B,T,C]
        if self.hparams.get('use_text_cond',True):
            # cond = style_embed
            h_ling = self.content_encoder(ph_token,cond,src_nonpadding.transpose(1,2))
        else:
            h_ling = self.content_encoder(ph_token)

        # add dur
        dur_inp = h_ling
        if hparams.get('use_style_adapter',False):
            dur_inp = self.pre_style_adpater(dur_inp,context=cond) * src_nonpadding
        if self.hparams['use_cond_predictor']:
            # cond = style_embed
            mel2ph = self.forward_dur(dur_inp,cond, mel2ph,ph_token, ret)
        else:
            mel2ph = self.forward_dur(dur_inp,None, mel2ph,ph_token, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(dur_inp, mel2ph) * tgt_nonpadding
   
        # add pitch embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = decoder_inp 
            if self.hparams['use_cond_predictor']:
                # cond = style_embed
                decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,cond, f0, uv, mel2ph, ret, h_ling)
            else:
                decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,None, f0, uv, mel2ph, ret, h_ling)

        if hparams.get('use_style_adapter',False):
            decoder_inp = self.post_style_adapter(decoder_inp,context=cond) * tgt_nonpadding    
        
        if self.hparams.get('use_adpative_cond_fft',False):
            decoder_inp = self.adaptive_encoder(decoder_inp,context=cond)
            
        # decoder input
        ### 暂时diff decoder只用spk_embed做cond
        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding
        # ret['decoder_inp'] = decoder_inp * tgt_nonpadding
        # ret['prompt'] = style_embed + spk_embed
        return ret
    
    def forward_spk_embed(self,ref_mel,style_embed,ret):
        if self.hparams.get('spk_mel_dim',20)==20:
            spk_embed = self.speaker_enc(ref_mel[:,:,:20]) # [B,C]
        else:
            spk_embed = self.speaker_enc(ref_mel)
        # spk_embed = self.spk_in(torch.cat((style_embed,spk_embed),dim=-1)).unsqueeze(dim=1)
        spk_embed = self.spk_in(spk_embed).unsqueeze(dim=1)
        spk_embed = spk_embed.transpose(0,1)   #[1, B, C]
        spk_bank = self.spk_bank
        spk_bank = spk_bank.repeat(spk_embed.shape[1],1,1).transpose(0,1)    #[N, C]  --> [N, B, C]
        spk_embed, attn_weight = self.spk_cross_attention(spk_embed, spk_bank, spk_bank)
        # ret['spk_attn_weight'] = attn_weight
        spk_embed = spk_embed.transpose(0,1)    #[B, 1, C]
        spk_embed = self.spk_out(spk_embed)
        ret['spk_embed'] = spk_embed.squeeze(dim=1)
        return spk_embed

    # def forward_style_embed(self, style_embed,ret):
    #     beta = 0.6
    #     bert_style_embed = style_embed
    #     style_embed = self.mlp(style_embed)  # [B,T,C]
    #     style_embed = beta*style_embed + (1-beta)*bert_style_embed
    #     style_embed = self.style_in(style_embed)
    #     padding_mask = (style_embed.sum(dim=-1)).eq(0).data  # [B,T]
    #     style_embed = style_embed.transpose(0,1)   # [T,B,C]
    #     style_embed, attn_weights = self.style_bank(style_embed,padding_mask=padding_mask)
    #     ret['attn_weights'] = attn_weights
    #     style_embed = style_embed.transpose(0,1)  # [B,T,C]
    #     return style_embed

    def forward_style_embed(self, style_embed,ret):
        style_embed = self.mlp(style_embed)  # [B,T,C]
        padding_mask = (style_embed.sum(dim=-1)).eq(0).data  # [B,T]
        style_embed = style_embed.transpose(0,1)   # [T,B,C]
        style_embed, attn_weights = self.style_bank(style_embed,padding_mask=padding_mask)
        ret['attn_weights'] = attn_weights
        style_embed = style_embed.transpose(0,1)  # [B,T,C]
        return style_embed
    
    def forward_dur(self, dur_input,cond, mel2ph, ph_token, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param ph_token: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = ph_token == 100
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        if cond is None:
            dur = self.dur_predictor(dur_input, src_padding)
        else: 
            dur = self.dur_predictor(dur_input,cond)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph

    def forward_pitch(self, decoder_inp,cond, f0, uv, mel2ph, ret, encoder_out=None):
        if self.hparams['pitch_type'] == 'frame':
            pitch_pred_inp = decoder_inp
            pitch_padding = mel2ph == 0
        else:
            pitch_pred_inp = encoder_out
            pitch_padding = encoder_out.abs().sum(-1) == 0
            uv = None
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        if cond is None:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        else:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp,cond)
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            pitch_padding=pitch_padding)
        if self.hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

class TTS_Multi_Stream_Network(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.hidden_size = hparams['hidden_size']
        # self.codebook = nn.Embedding(hparams['num_codes']+1,hparams['code_dim'],padding_idx=hparams['num_codes'])
        self.content_encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)
        ### speaker encoder
        self.multi_stream_encoder = MultiStream_Spk_Encoder(spk_bottleneck_dim=hparams['spk_bottleneck_dim'],\
            hidden_size=hparams['hidden_size'])
        
        ### style encoder
        bert_dim = hparams['bert_dim']
        fc = [EqualLinear(bert_dim, bert_dim)]
        for i in range(2):
            fc.append(EqualLinear(bert_dim, bert_dim))
        fc.append(EqualLinear(bert_dim, hparams['hidden_size']))
        # fc.append(EqualLinear(bert_dim, bert_dim))
        self.mlp = nn.Sequential(*fc)
        # self.style_in = nn.Linear(hparams['bert_dim'],hparams['hidden_size'],bias=False)
        self.style_bank = StyleBankExtractor(hparams['hidden_size'],hparams['sbank_layers'],hparams['sbank_size'],
                                               hparams['sbank_dim'],hparams['sbank_num_heads'])
        
        if hparams.get('use_style_adapter',False):
            print('*'*90)
            self.pre_style_adpater = ConditionableTransformer(
                dim = self.hidden_size,
                depth = 2,
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                ff_causal_conv = True,
                dim_cond_mult = None,
                use_flash = False,
                cross_attn = True
                )
            self.post_style_adapter = ConditionableTransformer(
                dim = self.hidden_size,
                depth = 2,
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                ff_causal_conv = True,
                dim_cond_mult = None,
                use_flash = False,
                cross_attn = True
                )


        if hparams.get('use_adpative_cond_fft',False):
            self.adaptive_encoder = ConditionableTransformer(
                dim = self.hidden_size,
                depth = 2,
                dim_head = 64,
                heads = 8,
                ff_mult = 4,
                ff_causal_conv = True,
                dim_cond_mult = None,
                use_flash = False,
                cross_attn = True
                )
        
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        if hparams['use_cond_predictor']:
            self.dur_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=1)
        else:
            self.dur_predictor = DurationPredictor(
                self.hidden_size,
                n_chans=predictor_hidden,
                n_layers=hparams['dur_predictor_layers'],
                dropout_rate=hparams['predictor_dropout'],
                kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            if hparams['use_cond_predictor']:
                self.pitch_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=2)
            else:
                self.pitch_predictor = PitchPredictor(
                    self.hidden_size, n_chans=predictor_hidden,
                    n_layers=5, dropout_rate=0.3, odim=2,
                    kernel_size=hparams['predictor_kernel'])

    def forward(self, mel,ref_mel,ph_token,style_embed,mel2ph,f0,uv,**kwargs):
        ret = {}
        # assert mel.shape[1]==mel_unit.shape[1],'error mel length'
        # assert drive_mel.shape[1]==dirve_mel_unit.shape[1],'error mel length'
        style_embed = self.forward_style_embed(style_embed,ret)
        spk_embed = self.forward_spk_embed(ref_mel,style_embed,ret)
        if self.hparams.get('add_cond',True):
            cond = style_embed + spk_embed  # [B,32,C]
            # cond = style_embed
        else:
            cond = torch.cat((style_embed,spk_embed),dim=1)
        
        src_nonpadding = (ph_token > 0).float()[:, :, None]   # [B,T,C]
        if self.hparams.get('use_text_cond',True):
            # cond = style_embed
            h_ling = self.content_encoder(ph_token,cond,src_nonpadding.transpose(1,2))
        else:
            h_ling = self.content_encoder(ph_token)

        # add dur
        dur_inp = h_ling
        if hparams.get('use_style_adapter',False):
            dur_inp = self.pre_style_adpater(dur_inp,context=cond) * src_nonpadding
        if self.hparams['use_cond_predictor']:
            # cond = style_embed
            mel2ph = self.forward_dur(dur_inp,cond, mel2ph,ph_token, ret)
        else:
            mel2ph = self.forward_dur(dur_inp,None, mel2ph,ph_token, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(dur_inp, mel2ph) * tgt_nonpadding
   
        # add pitch embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = decoder_inp 
            if self.hparams['use_cond_predictor']:
                # cond = style_embed
                decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,cond, f0, uv, mel2ph, ret, h_ling)
            else:
                decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,None, f0, uv, mel2ph, ret, h_ling)

        if hparams.get('use_style_adapter',False):
            decoder_inp = self.post_style_adapter(decoder_inp,context=cond) * tgt_nonpadding    
        
        if self.hparams.get('use_adpative_cond_fft',False):
            # decoder_inp = self.adaptive_encoder(decoder_inp,spk_embed)
            decoder_inp = self.adaptive_encoder(decoder_inp,context=cond)
            
        # decoder input
        ### 暂时diff decoder只用spk_embed做cond
        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding
        # ret['decoder_inp'] = decoder_inp * tgt_nonpadding
        # ret['prompt'] = style_embed + spk_embed
        return ret
    
    def forward_spk_embed(self,ref_mel,style_embed,ret):
        spk_embed = self.multi_stream_encoder(ref_mel)[:,None,:]  # [B,1,C]
        ret['spk_embed'] = spk_embed.squeeze(dim=1)
        return spk_embed

    # def forward_style_embed(self, style_embed,ret):
    #     beta = 0.6
    #     bert_style_embed = style_embed
    #     style_embed = self.mlp(style_embed)  # [B,T,C]
    #     style_embed = beta*style_embed + (1-beta)*bert_style_embed
    #     style_embed = self.style_in(style_embed)
    #     padding_mask = (style_embed.sum(dim=-1)).eq(0).data  # [B,T]
    #     style_embed = style_embed.transpose(0,1)   # [T,B,C]
    #     style_embed, attn_weights = self.style_bank(style_embed,padding_mask=padding_mask)
    #     ret['attn_weights'] = attn_weights
    #     style_embed = style_embed.transpose(0,1)  # [B,T,C]
    #     return style_embed

    def forward_style_embed(self, style_embed,ret):
        style_embed = self.mlp(style_embed)  # [B,T,C]
        padding_mask = (style_embed.sum(dim=-1)).eq(0).data  # [B,T]
        style_embed = style_embed.transpose(0,1)   # [T,B,C]
        style_embed, attn_weights = self.style_bank(style_embed,padding_mask=padding_mask)
        ret['attn_weights'] = attn_weights
        style_embed = style_embed.transpose(0,1)  # [B,T,C]
        return style_embed
    
    def forward_dur(self, dur_input,cond, mel2ph, ph_token, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param ph_token: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = ph_token == 100
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        if cond is None:
            dur = self.dur_predictor(dur_input, src_padding)
        else: 
            dur = self.dur_predictor(dur_input,cond)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph

    def forward_pitch(self, decoder_inp,cond, f0, uv, mel2ph, ret, encoder_out=None):
        if self.hparams['pitch_type'] == 'frame':
            pitch_pred_inp = decoder_inp
            pitch_padding = mel2ph == 0
        else:
            pitch_pred_inp = encoder_out
            pitch_padding = encoder_out.abs().sum(-1) == 0
            uv = None
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        if cond is None:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        else:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp,cond)
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            pitch_padding=pitch_padding)
        if self.hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

class TTC_Network(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.hidden_size = hparams['hidden_size']
        # self.codebook = nn.Embedding(hparams['num_codes']+1,hparams['code_dim'],padding_idx=hparams['num_codes'])
        self.content_encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)
        ### speaker encoder
        self.speaker_enc = ECAPA_TDNN_SMALL(hparams.get('spk_mel_dim',20),hparams['hidden_size'])
        # self.spk_in = nn.Linear(hparams['hidden_size']*2, hparams['spk_bsvt_dim'],bias=False)
        self.spk_in = nn.Linear(hparams['hidden_size'], hparams['spk_bsvt_dim'],bias=False)
        self.spk_cross_attention = MultiheadAttention(embed_dim=hparams['spk_bsvt_dim'], num_heads=2)
        self.spk_bank = torch.nn.Parameter(torch.rand((hparams['spk_bsvt_num'], hparams['spk_bsvt_dim'])))
        self.spk_out = nn.Linear(hparams['spk_bsvt_dim'], hparams['hidden_size'],bias=False)
        
        ### style encoder
        bert_dim = hparams['bert_dim']
        fc = [EqualLinear(bert_dim, bert_dim)]
        for i in range(2):
            fc.append(EqualLinear(bert_dim, bert_dim))
        fc.append(EqualLinear(bert_dim, hparams['hidden_size']))
        # fc.append(EqualLinear(bert_dim, bert_dim))
        self.mlp = nn.Sequential(*fc)
        # self.style_in = nn.Linear(hparams['bert_dim'],hparams['hidden_size'],bias=False)
        self.style_bank = StyleBankExtractor(hparams['hidden_size'],hparams['sbank_layers'],hparams['sbank_size'],
                                               hparams['sbank_dim'],hparams['sbank_num_heads'])
        
        # Mixstyle
        self.norm = MixStyle(p=0.5, alpha=0.1, eps=1e-6, hidden_size=self.hidden_size)
        self.adapter = ConditionableTransformer(
            dim = self.hidden_size,
            depth = 2,
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            ff_causal_conv = True,
            dim_cond_mult = None,
            use_flash = False,
            cross_attn = True
        )
        
        if hparams.get('use_adpative_cond_fft',False):
            self.adaptive_encoder = FS_DECODERS[hparams.get('adaptive_encoder_type','cond_fft')](hparams)
        
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        if hparams['use_cond_predictor']:
            self.dur_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=1)
        else:
            self.dur_predictor = DurationPredictor(
                self.hidden_size,
                n_chans=predictor_hidden,
                n_layers=hparams['dur_predictor_layers'],
                dropout_rate=hparams['predictor_dropout'],
                kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            if hparams['use_cond_predictor']:
                self.pitch_predictor = PitchPredictor(
                    self.hidden_size, n_chans=predictor_hidden,
                    n_layers=5, dropout_rate=0.3, odim=2,
                    kernel_size=hparams['predictor_kernel'])
                # self.pitch_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=2)
            else:
                self.pitch_predictor = PitchPredictor(
                    self.hidden_size, n_chans=predictor_hidden,
                    n_layers=5, dropout_rate=0.3, odim=2,
                    kernel_size=hparams['predictor_kernel'])
            self.pitch_agnostic_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=2)

    def forward(self, mel,ref_mel,ph_token,style_embed,mel2ph,f0,uv,**kwargs):
        ret = {}
        # assert mel.shape[1]==mel_unit.shape[1],'error mel length'
        # assert drive_mel.shape[1]==dirve_mel_unit.shape[1],'error mel length'
        style_embed = self.forward_style_embed(style_embed,ret)
        spk_embed = self.forward_spk_embed(ref_mel,style_embed,ret)
        if self.hparams.get('add_cond',True):
            cond = style_embed + spk_embed  # [B,32,C]
            # cond = style_embed
        else:
            cond = torch.cat((style_embed,spk_embed),dim=1)
        
        src_nonpadding = (ph_token > 0).float()[:, :, None]   # [B,T,C]
        if self.hparams.get('use_text_cond',True):
            # cond = style_embed
            h_ling = self.content_encoder(ph_token,cond,src_nonpadding.transpose(1,2))
        else:
            h_ling = self.content_encoder(ph_token)

        # add dur
        dur_inp = h_ling
        if self.hparams['use_cond_predictor']:
            # cond = style_embed
            mel2ph = self.forward_dur(dur_inp,cond, mel2ph,ph_token, ret)
        else:
            mel2ph = self.forward_dur(dur_inp,None, mel2ph,ph_token, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(h_ling, mel2ph) * tgt_nonpadding

        ### get speaker-specific features and speaker-agnostic 
        # decoder_inp = self.adapter(decoder_inp,context=cond) * tgt_nonpadding
        pitch_inp_spk_agnostic = self.norm(decoder_inp, spk_embed)* tgt_nonpadding 
        pitch_inp_spk_specific = decoder_inp + spk_embed
        
        
        # add pitch embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = decoder_inp 
            global_step = kwargs.get('global_step')
            if self.hparams['use_cond_predictor']:
                # cond = style_embed
                # decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,cond, f0, uv, mel2ph, ret, h_ling)
                decoder_inp = decoder_inp + self.forward_pitch(pitch_inp_spk_specific,pitch_inp_spk_agnostic,cond, f0, uv, mel2ph, ret,style_embed,h_ling,global_step=global_step)
            else:
                # decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,None, f0, uv, mel2ph, ret, h_ling)
                decoder_inp = decoder_inp + self.forward_pitch(pitch_inp_spk_specific,pitch_inp_spk_agnostic,None, f0, uv, mel2ph, ret,style_embed,h_ling,global_step=global_step)
            
        
        if self.hparams.get('use_adpative_cond_fft',False):
            decoder_inp = self.adaptive_encoder(decoder_inp,spk_embed)
            
        # decoder input
        ### 暂时diff decoder只用spk_embed做cond
        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding
        return ret
    
    def forward_spk_embed(self,ref_mel,style_embed,ret):
        if self.hparams.get('spk_mel_dim',20)==20:
            spk_embed = self.speaker_enc(ref_mel[:,:,:20]) # [B,C]
        else:
            spk_embed = self.speaker_enc(ref_mel)
        # spk_embed = self.spk_in(torch.cat((style_embed,spk_embed),dim=-1)).unsqueeze(dim=1)
        spk_embed = self.spk_in(spk_embed).unsqueeze(dim=1)
        spk_embed = spk_embed.transpose(0,1)   #[1, B, C]
        spk_bank = self.spk_bank
        spk_bank = spk_bank.repeat(spk_embed.shape[1],1,1).transpose(0,1)    #[N, C]  --> [N, B, C]
        spk_embed, attn_weight = self.spk_cross_attention(spk_embed, spk_bank, spk_bank)
        # ret['spk_attn_weight'] = attn_weight
        spk_embed = spk_embed.transpose(0,1)    #[B, 1, C]
        spk_embed = self.spk_out(spk_embed)
        ret['spk_embed'] = spk_embed.squeeze(dim=1)
        return spk_embed

    # def forward_style_embed(self, style_embed,ret):
    #     beta = 0.6
    #     bert_style_embed = style_embed
    #     style_embed = self.mlp(style_embed)  # [B,T,C]
    #     style_embed = beta*style_embed + (1-beta)*bert_style_embed
    #     style_embed = self.style_in(style_embed)
    #     padding_mask = (style_embed.sum(dim=-1)).eq(0).data  # [B,T]
    #     style_embed = style_embed.transpose(0,1)   # [T,B,C]
    #     style_embed, attn_weights = self.style_bank(style_embed,padding_mask=padding_mask)
    #     ret['attn_weights'] = attn_weights
    #     style_embed = style_embed.transpose(0,1)  # [B,T,C]
    #     return style_embed

    def forward_style_embed(self, style_embed,ret):
        style_embed = self.mlp(style_embed)  # [B,T,C]
        padding_mask = (style_embed.sum(dim=-1)).eq(0).data  # [B,T]
        style_embed = style_embed.transpose(0,1)   # [T,B,C]
        style_embed, attn_weights = self.style_bank(style_embed,padding_mask=padding_mask)
        ret['attn_weights'] = attn_weights
        style_embed = style_embed.transpose(0,1)  # [B,T,C]
        return style_embed
    
    def forward_dur(self, dur_input,cond, mel2ph, ph_token, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param ph_token: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = ph_token == 100
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        if cond is None:
            dur = self.dur_predictor(dur_input, src_padding)
        else: 
            dur = self.dur_predictor(dur_input,cond)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph
    
    def forward_pitch(self, pitch_inp_spk_specific,pitch_inp_spk_agnostic,cond, f0, uv, mel2ph, ret,spk_agnostic_cond,encoder_out=None,global_step=0):
        if self.hparams['pitch_type'] == 'frame':
            pitch_padding = mel2ph == 0
        if self.hparams['predictor_grad'] != 1:
            pitch_inp_spk_specific  = pitch_inp_spk_specific.detach() + \
                             self.hparams['predictor_grad'] * (pitch_inp_spk_specific - pitch_inp_spk_specific.detach())
            pitch_inp_spk_agnostic  = pitch_inp_spk_agnostic.detach() + \
                             self.hparams['predictor_grad'] * (pitch_inp_spk_agnostic - pitch_inp_spk_agnostic.detach())
        
        # print(pitch_inp_spk_agnostic.shape,'*'*50)
        ### 前2w步，pitch的学习只参考text prompt,学习到一定的pitch提示信息，进一步loss减少，需要从spk中参考一定的pitch信息
        pitch_spk_agnostic = self.pitch_agnostic_predictor(pitch_inp_spk_agnostic,spk_agnostic_cond)
        if self.hparams['text_prompt_pitch_warmsup']>=global_step:
            pitch_pred = pitch_spk_agnostic
        else:
            pitch_spk_specific = self.pitch_predictor(pitch_inp_spk_specific)
            pitch_pred = pitch_spk_agnostic + pitch_spk_specific
        ret['pitch_pred'] = pitch_pred
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            pitch_padding=pitch_padding)
        if self.hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed

class Codec_WavLM_Network(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.hidden_size = hparams['hidden_size']
        # self.codebook = nn.Embedding(hparams['num_codes']+1,hparams['code_dim'],padding_idx=hparams['num_codes'])
        self.content_encoder = FS_ENCODERS[hparams['encoder_type']](hparams, dict_size)
        ### speaker encoder
        self.spk_enc = WavLMSpeakerAdapter(spk_bottleneck_dim=hparams['spk_bottleneck_dim'])
        
        ### style encoder
        bert_dim = hparams['bert_dim']
        fc = [EqualLinear(bert_dim, bert_dim)]
        for i in range(2):
            fc.append(EqualLinear(bert_dim, bert_dim))
        fc.append(EqualLinear(bert_dim, hparams['hidden_size']))
        # fc.append(EqualLinear(bert_dim, bert_dim))
        self.mlp = nn.Sequential(*fc)
        # self.style_in = nn.Linear(hparams['bert_dim'],hparams['hidden_size'],bias=False)
        self.style_bank = StyleBankExtractor(hparams['hidden_size'],hparams['sbank_layers'],hparams['sbank_size'],
                                               hparams['sbank_dim'],hparams['sbank_num_heads'])
        
        if hparams.get('use_adpative_cond_fft',False):
            self.adaptive_encoder = FS_DECODERS[hparams.get('adaptive_encoder_type','cond_fft')](hparams)
        
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        if hparams['use_cond_predictor']:
            self.dur_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=1)
        else:
            self.dur_predictor = DurationPredictor(
                self.hidden_size,
                n_chans=predictor_hidden,
                n_layers=hparams['dur_predictor_layers'],
                dropout_rate=hparams['predictor_dropout'],
                kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            if hparams['use_cond_predictor']:
                self.pitch_predictor = DurationPitchPredictor(dim=predictor_hidden,depth=hparams['pred_depth'],out_dim=2)
            else:
                self.pitch_predictor = PitchPredictor(
                    self.hidden_size, n_chans=predictor_hidden,
                    n_layers=5, dropout_rate=0.3, odim=2,
                    kernel_size=hparams['predictor_kernel'])

    def forward(self, mel,ref_mel,ph_token,style_embed,wavlm_spk_embed,mel2ph,f0,uv,**kwargs):
        ret = {}
        # assert mel.shape[1]==mel_unit.shape[1],'error mel length'
        # assert drive_mel.shape[1]==dirve_mel_unit.shape[1],'error mel length'
        style_embed = self.forward_style_embed(style_embed,ret)
        spk_embed = self.forward_spk_embed(wavlm_spk_embed,ret)
        if self.hparams.get('add_cond',True):
            cond = style_embed + spk_embed  # [B,32,C]
            # cond = style_embed
        else:
            cond = torch.cat((style_embed,spk_embed),dim=1)
        
        src_nonpadding = (ph_token > 0).float()[:, :, None]   # [B,T,C]
        if self.hparams.get('use_text_cond',True):
            # cond = style_embed
            h_ling = self.content_encoder(ph_token,cond,src_nonpadding.transpose(1,2))
        else:
            h_ling = self.content_encoder(ph_token)

        # add dur
        dur_inp = h_ling
        if self.hparams['use_cond_predictor']:
            # cond = style_embed
            mel2ph = self.forward_dur(dur_inp,cond, mel2ph,ph_token, ret)
        else:
            mel2ph = self.forward_dur(dur_inp,None, mel2ph,ph_token, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(h_ling, mel2ph) * tgt_nonpadding
   
        # add pitch embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = decoder_inp 
            if self.hparams['use_cond_predictor']:
                # cond = style_embed
                decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,cond, f0, uv, mel2ph, ret, h_ling)
            else:
                decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,None, f0, uv, mel2ph, ret, h_ling)
            
        
        if self.hparams.get('use_adpative_cond_fft',False):
            decoder_inp = self.adaptive_encoder(decoder_inp,spk_embed)
            
        # decoder input
        ### 暂时diff decoder只用spk_embed做cond
        ret['decoder_inp'] = decoder_inp = (decoder_inp + spk_embed) * tgt_nonpadding
        # ret['decoder_inp'] = decoder_inp * tgt_nonpadding
        # ret['prompt'] = style_embed + spk_embed
        return ret
    
    def forward_spk_embed(self,wavlm_spk_embed,ret):
        spk_embed = self.spk_enc(wavlm_spk_embed)[:,None,:]  # [B,1,C]
        ret['spk_embed'] = spk_embed.squeeze(dim=1)
        return spk_embed

    # def forward_style_embed(self, style_embed,ret):
    #     beta = 0.6
    #     bert_style_embed = style_embed
    #     style_embed = self.mlp(style_embed)  # [B,T,C]
    #     style_embed = beta*style_embed + (1-beta)*bert_style_embed
    #     style_embed = self.style_in(style_embed)
    #     padding_mask = (style_embed.sum(dim=-1)).eq(0).data  # [B,T]
    #     style_embed = style_embed.transpose(0,1)   # [T,B,C]
    #     style_embed, attn_weights = self.style_bank(style_embed,padding_mask=padding_mask)
    #     ret['attn_weights'] = attn_weights
    #     style_embed = style_embed.transpose(0,1)  # [B,T,C]
    #     return style_embed

    def forward_style_embed(self, style_embed,ret):
        style_embed = self.mlp(style_embed)  # [B,T,C]
        padding_mask = (style_embed.sum(dim=-1)).eq(0).data  # [B,T]
        style_embed = style_embed.transpose(0,1)   # [T,B,C]
        style_embed, attn_weights = self.style_bank(style_embed,padding_mask=padding_mask)
        ret['attn_weights'] = attn_weights
        style_embed = style_embed.transpose(0,1)  # [B,T,C]
        return style_embed
    
    def forward_dur(self, dur_input,cond, mel2ph, ph_token, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param ph_token: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = ph_token == 100
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        if cond is None:
            dur = self.dur_predictor(dur_input, src_padding)
        else: 
            dur = self.dur_predictor(dur_input,cond)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph

    def forward_pitch(self, decoder_inp,cond, f0, uv, mel2ph, ret, encoder_out=None):
        if self.hparams['pitch_type'] == 'frame':
            pitch_pred_inp = decoder_inp
            pitch_padding = mel2ph == 0
        else:
            pitch_pred_inp = encoder_out
            pitch_padding = encoder_out.abs().sum(-1) == 0
            uv = None
        if self.hparams['predictor_grad'] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + \
                             self.hparams['predictor_grad'] * (pitch_pred_inp - pitch_pred_inp.detach())
        if cond is None:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp)
        else:
            ret['pitch_pred'] = pitch_pred = self.pitch_predictor(pitch_pred_inp,cond)
        use_uv = self.hparams['pitch_type'] == 'frame' and self.hparams['use_uv']
        if f0 is None:
            f0 = pitch_pred[:, :, 0]
            if use_uv:
                uv = pitch_pred[:, :, 1] > 0
        f0_denorm = denorm_f0(f0, uv if use_uv else None, pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0 [B, T_txt]
        ret['f0_denorm'] = f0_denorm
        ret['f0_denorm_pred'] = denorm_f0(
            pitch_pred[:, :, 0], (pitch_pred[:, :, 1] > 0) if use_uv else None,
            pitch_padding=pitch_padding)
        if self.hparams['pitch_type'] == 'ph':
            pitch = torch.gather(F.pad(pitch, [1, 0]), 1, mel2ph)
            ret['f0_denorm'] = torch.gather(F.pad(ret['f0_denorm'], [1, 0]), 1, mel2ph)
            ret['f0_denorm_pred'] = torch.gather(F.pad(ret['f0_denorm_pred'], [1, 0]), 1, mel2ph)
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed