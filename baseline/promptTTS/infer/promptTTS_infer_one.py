import os
import numpy as np
import torch
import sys
import shutil
import pandas as pd
from tqdm import tqdm

sys.path.append('')

from utils.audio import librosa_wav2spec
from utils.commons.hparams import hparams
from utils.commons.ckpt_utils import load_ckpt
from infer.base_tts_infer import BaseTTSInfer
from modules.tts.promptTTS.promptts import PromptTTS
from data_gen.tts.base_preprocess import BasePreprocessor
from transformers import AutoTokenizer, AutoModel
from infer.load_fintune_bert_style import Bert_Style_Finetune

class TCTTSInfer(BaseTTSInfer):
    def __init__(self, hparams, device=None):
        if device is None:
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = 'cpu'
        self.hparams = hparams
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.preprocessor = BasePreprocessor()
        self.ph_encoder, self.word_encoder = self.preprocessor.load_dict(self.data_dir)
        self.model = self.build_model()
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)
        self.adapter_net = self.build_adapter()

    def build_adapter(self):
        adapter_net = Bert_Style_Finetune()
        load_ckpt(adapter_net, 'checkpoints/finetune_bert_style', 'model')
        adapter_net.to(self.device)
        adapter_net.eval()
        return adapter_net

    def build_model(self):
        dict_size = len(self.ph_encoder)
        model = PromptTTS(dict_size,hparams)
        load_ckpt(model, hparams['work_dir'], 'model')
        model.to(self.device)
        model.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        ph_token = sample['ph_tokens']
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']
        style_embed = sample['style_embed']

        with torch.no_grad():
            output = self.model(ph_token,style_embed,mel2ph,f0,uv,infer=True)
            mel_out = output['mel_out']
            gen_wav = self.run_vocoder(mel_out, output['f0_denorm'])
            gen_wav = gen_wav.cpu().numpy()[0]

        return gen_wav
    
    def run_vocoder(self, c, f0):
        c = c.transpose(2, 1)
        y = self.vocoder(c, f0)[:, 0]
        return y
    
    def infer_once(self, inp):
        inp = self.preprocess_input(inp)
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    def preprocess_input(self, inp):    
        preprocessor = self.preprocessor
        text_raw = inp['text']
        ph, txt, words, ph2word, ph_gb_word = preprocessor.txt_to_ph(preprocessor.txt_processor, text_raw)
        ph_token = self.ph_encoder.encode(ph)

        ph_token = ph_token
        mel2ph = inp['mel2ph']
        f0 = inp['f0']
        uv = inp['uv']
        style_embed = inp['style_embed']

        item = {'ph_token': ph_token,'mel2ph': mel2ph,'f0':f0,'uv':uv,'style_embed':style_embed}
        return item

    def input_to_batch(self, item):   
        ph_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        mel2ph = None
        f0 = None
        uv = None
        style_embed = torch.Tensor(item['style_embed'])[None, :].to(self.device)

        batch = {'ph_tokens': ph_tokens,
                'mel2ph': mel2ph,
                'f0':f0,
                'uv':uv,
                'style_embed':style_embed}
        return batch
    
    def get_fintune_bert_embed(self,style_embed):
        with torch.no_grad():
            output = self.adapter_net(style_embed.to(self.device))
            y = output['pooling_embed'].squeeze(dim=0)
            return y

    @classmethod
    def example_run(cls,raw_text,prompt_text):
        from utils.commons.hparams import set_hparams
        from utils.commons.hparams import hparams as hp
        from utils.audio.io import save_wav
        import glob
        import shutil
        dict = set_hparams()
        infer_ins = cls(hp,'cuda')
        exp_name = dict['exp_name']
        base_dir = f'infer/one_output'
        os.makedirs(base_dir, exist_ok=True)


        style_embed = get_style_embed(prompt_text)
        style_embed = infer_ins.get_fintune_bert_embed(style_embed.unsqueeze(dim=0))
        inp = {
            # 'ph_token': ph_token,
            'style_embed': style_embed,
            'mel2ph':None,
            'f0':None,
            'uv':None,
            'text': raw_text
        }
        gen_wav = infer_ins.infer_once(inp)
        save_wav(gen_wav, f'{base_dir}/gen.wav', hp['audio_sample_rate'])

tokenizer = AutoTokenizer.from_pretrained("../ckpt/bert-base-uncased")
model = AutoModel.from_pretrained("../ckpt/bert-base-uncased")      
def get_style_embed(style_prompt):
    inputs = tokenizer(style_prompt, return_tensors="pt")
    outputs = model(**inputs)
    return outputs[0][0]

if __name__ == '__main__':
    raw_text = 'Do you wish me to have a fire lighted in the alcove?'
    prompt_text = "Engaging in dialogue, the speaker's pitch is neither too high nor too low."
    TCTTSInfer.example_run(raw_text,prompt_text)