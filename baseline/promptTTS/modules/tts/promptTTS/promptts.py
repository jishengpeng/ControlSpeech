from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from utils.commons.hparams import hparams
from modules.commons.layers import Embedding
from modules.commons.rel_transformer import ConditionalRelTransformerEncoder
from modules.commons.transformer import ConditionalFastSpeechDecoder
from modules.commons.nar_tts_modules import PitchPredictor, DurationPredictor, LengthRegulator
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states

    
class PromptTTS(nn.Module):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.enc_layers = hparams['enc_layers']
        self.hidden_size = hparams['hidden_size']
        self.content_encoder = ConditionalRelTransformerEncoder(
            dict_size, hparams['hidden_size'], hparams['hidden_size'],
            hparams['ffn_hidden_size'], hparams['num_heads'], hparams['enc_layers'],
            hparams['enc_ffn_kernel_size'], hparams['dropout'], prenet=hparams['enc_prenet'], pre_ln=hparams['enc_pre_ln'])
        self.decoder = ConditionalFastSpeechDecoder(
            hparams['hidden_size'], hparams['dec_layers'], hparams['dec_ffn_kernel_size'], hparams['num_heads'])
        self.mel_out = nn.Linear(self.hidden_size, 80, bias=True)
        
        predictor_hidden = hparams['predictor_hidden'] if hparams['predictor_hidden'] > 0 else self.hidden_size
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=predictor_hidden,
            n_layers=hparams['dur_predictor_layers'],
            dropout_rate=hparams['predictor_dropout'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size, n_chans=predictor_hidden,
                n_layers=5, dropout_rate=0.3, odim=2,
                kernel_size=hparams['predictor_kernel'])
            

    def forward(self, ph_token,style_embed,mel2ph,f0,uv,**kwargs):
        ret = {}
        ### style_embed [B,C]
        
        src_nonpadding = (ph_token > 0).float()[:, :, None]   # [B,T,C]
        h_ling = self.content_encoder(ph_token,style_embed.unsqueeze(dim=1))

        # add dur
        dur_inp = h_ling
        mel2ph = self.forward_dur(dur_inp,None, mel2ph,ph_token, ret)
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        decoder_inp = expand_states(dur_inp, mel2ph) * tgt_nonpadding
   
        # add pitch embed
        if self.hparams['use_pitch_embed']:
            pitch_inp = decoder_inp 
            decoder_inp = decoder_inp + self.forward_pitch(pitch_inp,None, f0, uv, mel2ph, ret, h_ling)  
        
        x = self.decoder(decoder_inp,style_embed.unsqueeze(dim=1))
        x = self.mel_out(x)
        ret['mel_out'] = x
        return ret
    
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