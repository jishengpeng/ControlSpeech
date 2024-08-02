# ControlSpeech: Towards Simultaneous Zero-shot Speaker Cloning and Zero-shot Language Style Control With Decoupled Codec


[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.01205)
[![demo](https://img.shields.io/badge/ControlSpeech-Demo-red)](https://controlspeech.github.io)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20ControlSpeech-Models(Baseline)-blue)](https://drive.google.com/drive/folders/1H8U165KjLV05axwRWgZRsBdGO4R9T7F_?usp=drive_link)


## 🔥 News
- *2024.06*: We release ControlSpeech on arxiv and opensource ControlToolkit. The ControlToolkit primarily comprises three components: the VccmDataset(which is based on the TextrolSpeech dataset), a set of relevant evaluation metrics, and the replication code for related baselines.
- *2023.12*: [Textrolspeech](https://github.com/jishengpeng/TextrolSpeech) is accepted by ICASSP 2024.


## 1.VccmDataset
Speech:

- https://drive.google.com/file/d/1kNjYBqv_DohG8N3wF-J7kCSBmLxvs77N/view?usp=drive_link
- https://drive.google.com/file/d/1W9DuRsQuP3tfWwxFo0Dx8-Rg-XbGCIzH/view?usp=sharing
- http://www.openslr.org/60/

Text:

- The directory for storing text and related prompts is located in the "./VccmDataset" .

- The test set comprises four groups, corresponding to experiment one through experiment four as described in the paper.

## 2.Evaluation Metrics

### 2.1 Speed Acc
```shell
# To obtain the relevant grid, it is necessary to perform MFA alignment in advance
mfa align --clean $BASE_DIR/$MFA_INPUTS $BASE_DIR/mfa_dict.txt $BASE_DIR_AGO/model/$MODEL_NAME.zip  $BASE_DIR/$MFA_OUTPUTS 
python ./pitch_energy_dur_acc/duration_acc.py
```

### 2.2 Pitch and Energy Acc
```shell
python ./pitch_energy_dur_acc/energy_pitch_acc.py
```

### 2.3 Emotion Acc
```shell
# extract the embeddings of the test dataset using the pre-trained emotion2Vec model
python ./emotion_acc/emotion2vec_demo.py
# perform inference with the fine-tuned emotion model
python ./emotion_acc/controlnet_emo_acc.py  
```
### 2.4 Wer and SPK-SV
```shell
# wer based on whisper
python ./wer.py
# spk based on wavlm-sv
python ./spk_sv.py
```

## 3.Baseline(support PromptTTS and PromptStyle)


### 3.1 Installation

```shell
git clone https://github.com/jishengpeng/ControlSpeech.git && cd baseline
conda create --name controlspeech python==3.9
conda activate controlspeech
pip install -r requirements.txt
```

### 3.2 Usage

3.2.1. download baseline checkpoint from: [Google Drive](https://drive.google.com/drive/folders/1H8U165KjLV05axwRWgZRsBdGO4R9T7F_?usp=drive_link)

3.2.2. inference with arbitrary content text and prompt text

   ```shell
   #### promptTTS
   cd promptTTS 
   CUDA_VISIBLE_DEVICES=0 python infer/promptTTS_infer.py  --exp_name promptts1_style_baseline
   CUDA_VISIBLE_DEVICES=0 python infer/promptTTS_infer_one.py  --exp_name promptts1_style_baseline
   
   #### promptStyle 
   cd promptStyle & python inference.py
   ```


3.2.3. training for text-prompt tts models

   ```shell
   #### PromptTTS
   #1 data preprocessing, please refer to promptTTS/libritts_preprocess 
   #2 training
   CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/promptts.yaml  --exp_name promptts1_style_baseline --reset --hparams='seed=1200'
   
   #### PromptStyle
   #1 data preprocessing
   python preprocess.py --text_index 2 --filelists filelists/prompt_audio_sid_text_train_filelist.txt filelists/prompt_audio_sid_text_val_filelist.txt filelists/prompt_audio_sid_text_test_filelist.txt
   #2 training
   # use_style_encoder=False
   CUDA_VISIBLE_DEVICES=0 python train_ms_stage1.py -c configs/prompt.json -m promptStyle_baseline_s1
   # use_style_encoder=True
   CUDA_VISIBLE_DEVICES=0 python train_ms_stage2.py -c configs/prompt.json -m promptStyle_baseline
   ```


## Citation

If this code or VccmDataset contributes to your research, please cite our work:

```
@inproceedings{ji2024textrolspeech,
  title={Textrolspeech: A text style control speech corpus with codec language text-to-speech models},
  author={Ji, Shengpeng and Zuo, Jialong and Fang, Minghui and Jiang, Ziyue and Chen, Feiyang and Duan, Xinyu and Huai, Baoxing and Zhao, Zhou},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={10301--10305},
  year={2024},
  organization={IEEE}
}

@misc{ji2024controlspeech,
      title={ControlSpeech: Towards Simultaneous Zero-shot Speaker Cloning and Zero-shot Language Style Control With Decoupled Codec}, 
      author={Shengpeng Ji and Jialong Zuo and Minghui Fang and Siqi Zheng and Qian Chen and Wen Wang and Ziyue Jiang and Hai Huang and Xize Cheng and Rongjie Huang and Zhou Zhao},
      year={2024},
      eprint={2406.01205},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
