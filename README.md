# ControlSpeech: Towards Simultaneous Zero-shot Speaker Cloning and Zero-shot Language Style Control With Decoupled Codec


<!-- [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2402.12208.pdf) -->
[![demo](https://img.shields.io/badge/ControlSpeech-Demo-red)](https://controlspeech.github.io)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20ControlSpeech-Models(Baseline)-blue)](https://drive.google.com/drive/folders/1H8U165KjLV05axwRWgZRsBdGO4R9T7F_?usp=drive_link)

## Installation

```shell
git clone https://github.com/jishengpeng/ControlSpeech.git && cd baseline
conda create --name test python==3.9
conda activate test
pip install -r requirements.txt
```

## Usage

1. download baseline checkpoint from: [Google Drive](https://drive.google.com/drive/folders/1H8U165KjLV05axwRWgZRsBdGO4R9T7F_?usp=drive_link)

2. inference with arbitrary content text and prompt text

   ```shell
   #### promptTTS
   cd promptTTS 
   CUDA_VISIBLE_DEVICES=0 python infer/promptTTS_infer.py  --exp_name promptts1_style_baseline
   CUDA_VISIBLE_DEVICES=0 python infer/promptTTS_infer_one.py  --exp_name promptts1_style_baseline
   
   #### promptStyle 
   cd promptStyle & python inference.py
   ```


3. training for text-prompt tts models

   ```shell
   #### promptTTS
   #1 data preprocessing, please refer to promptTTS/libritts_preprocess 
   #2 training
   CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/promptts.yaml  --exp_name promptts1_style_baseline --reset --hparams='seed=1200'
   
   #### promptStyle
   #1 data preprocessing
   python preprocess.py --text_index 2 --filelists filelists/prompt_audio_sid_text_train_filelist.txt filelists/prompt_audio_sid_text_val_filelist.txt filelists/prompt_audio_sid_text_test_filelist.txt
   #2 training
   # use_style_encoder=False
   CUDA_VISIBLE_DEVICES=0 python train_ms_stage1.py -c configs/prompt.json -m promptStyle_baseline_s1
   # use_style_encoder=True
   CUDA_VISIBLE_DEVICES=0 python train_ms_stage2.py -c configs/prompt.json -m promptStyle_baseline
   ```
