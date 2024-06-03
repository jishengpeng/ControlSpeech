import os
import json
import glob
# from tqdm import tqdm
# from textgrid import TextGrid
import torch
import numpy as np
import sys
import pandas as pd
import warnings
from mel_process import librosa_wav2spec

text2mel_params = {'fft_size': 1024, 'hop_size': 320, 'win_size': 1024,
                    'audio_num_mel_bins': 80, 'fmin': 55, 'fmax': 7600,
                    'f0_min': 80, 'f0_max': 600, 'pitch_extractor': 'parselmouth',
                    'audio_sample_rate': 24000, 'loud_norm': False,
                    'mfa_min_sil_duration': 0.1, 'trim_eos_bos': False,
                    'with_align': True, 'text2mel_params': False,
                    'dataset_name': None,
                    'with_f0': True, 'min_mel_length': 64}

def parselmouth_pitch(wav_data, hop_size, audio_sample_rate, f0_min, f0_max,
                      voicing_threshold=0.6, *args, **kwargs):
    import parselmouth
    time_step = hop_size / audio_sample_rate * 1000
    n_mel_frames = int(len(wav_data) // hop_size)
    f0_pm = parselmouth.Sound(wav_data, audio_sample_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=voicing_threshold,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
    pad_size = (n_mel_frames - len(f0_pm) + 1) // 2
    f0 = np.pad(f0_pm, [[pad_size, n_mel_frames - len(f0_pm) - pad_size]], mode='constant')
    return f0

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

def process_pitch(mel, text2mel_params, wav):
    f0 = parselmouth_pitch(wav,text2mel_params['hop_size'], text2mel_params['audio_sample_rate'],
                        f0_min=text2mel_params['f0_min'], f0_max=text2mel_params['f0_max'])
    assert len(mel) == len(f0), (len(mel), len(f0))
    return f0

# def get_dur(tg_fn):
#     sec = 0.0
#     itvs = TextGrid.fromFile(tg_fn)[1]
#     ph_num = 0
#     for i in range(len(itvs)):
#         if itvs[i].mark=='':
#             continue
#         ph_num += 1
#         sec += itvs[i].maxTime - itvs[i].minTime
#         # if itvs[i].maxTime - itvs[i].minTime
#     return sec/ph_num

def get_pitch_energy(wav_fn):
    wav,mel = process_audio(wav_fn)
    f0 = process_pitch(mel,text2mel_params,wav)
    energy = (torch.from_numpy(mel).exp() ** 2).sum(-1).sqrt()
    energy = energy.numpy()
    assert energy.shape[0]==f0.shape[0],'Error!!!'
    unmask = (f0!=0).astype(int)
    num = sum(unmask)
    m_pitch = sum(f0*unmask)/num
    m_energy = sum(energy*unmask)/num
    return m_pitch,m_energy

# ### libritts所有数据的mfa结果在,spk_name是
# tg_fn = f"./data/processed/libritts_24k/mfa_outputs/{spk_name}/{item_name}.TextGrid"

def judge_energy(energy,target):
    if(energy < 1.6302763303833692 and target=="low"):
        return True
    if(energy > 2.363755779274812 and target=="high"):
        return True
    if(energy >= 1.6302763303833692 and energy <= 2.363755779274812 and target=="normal"):
        return True
    return False

def judge_pitch(pitch, target, gender):
    if(gender=="M"):
        if(pitch < 108.22490656727105 and target=="low"):
            return True
        if(pitch > 167.1542149675488 and target=="high"):
            return True
        if(pitch >= 108.22490656727105 and pitch <= 167.1542149675488 and target=="normal"):
            return True
        return False
    elif(gender=="F"):
        if(pitch < 179.21624321369268 and target=="low"):
            return True
        if(pitch > 252.1001267597759 and target=="high"):
            return True
        if(pitch >= 179.21624321369268 and pitch <= 252.1001267597759 and target=="normal"):
            return True
        return False
    else:
        assert 0, "Wrong"



def main():

    audio_path_list =["./test_gt_codec_undomainstyle"]
   
    for audio_path in audio_path_list:

        audio = os.listdir(audio_path)

        audio.sort()

        csv_path = "./libri_test_new_undomain_style_addid.csv"

        df = pd.read_csv(csv_path)

        energy_dict={}
        pitch_dict={}
        gender_dict={}

        id = df["new_id"]
        energy = df["energy"]
        pitch = df["pitch"]
        gender = df["gender"]


        for i in range(len(id)):
            energy_dict[str(id[i])]=energy[i]
            pitch_dict[str(id[i])]=pitch[i]
            gender_dict[str(id[i])]=gender[i]
        
        energy_right = 0

        pitch_right = 0

        cal_num = len(audio)


        for i in range(cal_num):
            tmp_id = str(audio[i].split('.')[0])

            m_pitch, m_energy = get_pitch_energy(os.path.join(audio_path,audio[i]))

            # print(tmp_id,m_energy,energy_dict[tmp_id])
            # if(gender_dict[tmp_id]=="M"):
            #     print(tmp_id, m_pitch, pitch_dict[tmp_id], gender_dict[tmp_id])


            if(judge_energy(m_energy,energy_dict[tmp_id])):
                energy_right+=1
                # print("right",tmp_id,m_energy,energy_dict[tmp_id])
            else:
                # print("wrong",tmp_id,m_energy,energy_dict[tmp_id])
                continue

            if(judge_pitch(m_pitch,pitch_dict[tmp_id],gender_dict[tmp_id])):
                pitch_right+=1
            else:
                # print("wrong",tmp_id, m_pitch, pitch_dict[tmp_id], gender_dict[tmp_id])
                continue

            # print(energy_right)
            # print(pitch_right)

        print("id",audio_path)
        print("energy_acc",energy_right/cal_num)
        print("pitch_acc",pitch_right/cal_num)



if __name__ == "__main__":
    main()


