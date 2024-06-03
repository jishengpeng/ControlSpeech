import os
import json
import glob
from tqdm import tqdm
from textgrid import TextGrid
import torch
import numpy as np
import sys
import warnings
from mel_process import librosa_wav2spec
import pandas as pd
import csv

text2mel_params = {'fft_size': 1024, 'hop_size': 320, 'win_size': 1024,
                    'audio_num_mel_bins': 80, 'fmin': 55, 'fmax': 7600,
                    'f0_min': 80, 'f0_max': 600, 'pitch_extractor': 'parselmouth',
                    'audio_sample_rate': 24000, 'loud_norm': False,
                    'mfa_min_sil_duration': 0.1, 'trim_eos_bos': False,
                    'with_align': True, 'text2mel_params': False,
                    'dataset_name': None,
                    'with_f0': True, 'min_mel_length': 64}



def process_audio(wav_fn):
    wav2spec_dict = librosa_wav2spec(
        wav_fn,
        fft_size=text2mel_params['fft_size'],
        hop_size=text2mel_params['hop_size'],
        win_length=text2mel_params['win_size'],
        num_mels=text2mel_params['audio_num_mel_bins'],
        fmin=text2mel_params['fmin'],
        fmax=text2mel_params['fmax'],
        sample_rate=text2mel_params['audio_sample_rate'],
        loud_norm=text2mel_params['loud_norm'])
    mel = wav2spec_dict['mel']
    wav = wav2spec_dict['wav'].astype(np.float16)
    return wav, mel



def get_dur(tg_fn):

    sec = 0.0
    itvs = TextGrid.fromFile(tg_fn)[1]
    ph_num = 0
    for i in range(len(itvs)):
        if itvs[i].mark=='':
            continue
        ph_num += 1
        sec += itvs[i].maxTime - itvs[i].minTime
        # if itvs[i].maxTime - itvs[i].minTime
    if(ph_num == 0):
        return 0

    return sec/ph_num

def judge_duration(duration,target):

    if(duration==0):
        print("有一些特殊的Grid")
        return True

    if(duration < 0.07269938650306747 and target=="high"):
        return True
    if(duration > 0.110666666666666 and target=="low"):
        return True
    if(duration >= 0.07269938650306747 and duration <= 0.110666666666666 and target=="normal"):
        return True
    return False


csv_path = "./libri_test_new_undomain_style_addid.csv"
df = pd.read_csv(csv_path)

id_duration_dict = {}

for i in range(len(df["new_id"])):
    id_duration_dict[df["new_id"][i]] = df["dur"][i]

# id = df["new_id"]
# duration_gt = df["dur"]

textgrid_path = "./output"

aa = glob.glob(os.path.join(textgrid_path,"*/*.TextGrid"))

assert len(aa) == len(df.values)

sum_duration = 0

for i in range(len(aa)):
    # breakpoint() 
    if(judge_duration(get_dur(aa[i]),id_duration_dict[aa[i].split('.')[0].split('/')[-1]])):
        sum_duration+=1
    
    print(sum_duration)

print("final",sum_duration/len(aa))



