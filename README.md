# ControlSpeech

## Installation

```shell
git clone https://github.com/jishengpeng/ControlSpeech.git && cd baseline
conda create --name test python==3.9
conda activate test
pip install -r requirements.txt
```

### Usage

1. download baseline checkpoint from: [Google Drive](https://drive.google.com/drive/folders/1H8U165KjLV05axwRWgZRsBdGO4R9T7F_?usp=drive_link)

2. inference with arbitrary content text and prompt text

   ```shell
   #### promptStyle 
   cd promptStyle & python inference.py
   #### promptTTS
   cd promptTTS 
   CUDA_VISIBLE_DEVICES=0 python infer/promptTTS_infer.py  --exp_name promptts1_style_baseline
   CUDA_VISIBLE_DEVICES=0 python infer/promptTTS_infer_one.py  --exp_name promptts1_style_baseline
   ```

   