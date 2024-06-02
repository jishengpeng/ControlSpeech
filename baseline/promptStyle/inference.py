import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from utils import load_wav_to_torch, load_filepaths_and_text
from mel_processing import spectrogram_torch,spec_to_mel_torch
from scipy.io.wavfile import write
import scipy.io.wavfile as wavfile
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("../ckpt/bert-base-uncased")
model = AutoModel.from_pretrained("../ckpt/bert-base-uncased")      
def get_style_embed(style_prompt):
    inputs = tokenizer(style_prompt, return_tensors="pt")
    outputs = model(**inputs)
    return outputs[-1]


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm
def get_mel(filename):
    audio, sampling_rate = load_wav_to_torch(filename)
    print(sampling_rate)
    audio_norm = audio / 32768.0
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(audio_norm, 1024, 24000, 320, 1024,center=False)
    spec = torch.squeeze(spec, 0)
    ### 提取梅尔特征
    mel = spec_to_mel_torch(spec, 1024, 80, 24000, 0.0, None)
    return mel
def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  print('222222222222222222222222')
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      print("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  print("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


device = 'cuda:0'
hps = utils.get_hparams_from_file("./configs/prompt.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).cuda(device=device)
_ = net_g.eval()
_ = utils.load_checkpoint("./logs/prompt_style_baseline/G_1365000.pth", net_g, None)

### synthesis
stn_tst = get_text("But it is not with a view to distinction that you should cultivate this talent, if you consult your own happiness.", hps)
style_prompt = "A speaker with low energy, and with slow speed."
style_embed = get_style_embed(style_prompt)
with torch.no_grad():
    x_tst = stn_tst.cuda(device=device).unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda(device=device)
    sid = torch.LongTensor([7]).cuda(device=device)
    style_embed = style_embed.cuda(0)
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid,mel=None,style_embed=style_embed, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    print(audio.shape)

# 保存为.wav文件
output_wav_file = "./output_audio.wav"
wavfile.write(output_wav_file, hps.data.sampling_rate, audio)