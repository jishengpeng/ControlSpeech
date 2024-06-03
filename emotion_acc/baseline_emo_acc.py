import os
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
import glob
import numpy as np 

from emotion_net import Emotion2Vec_finetume
from emotion_resnet import Emotion_Recognizer_ResNet
from ckpt_utils import load_ckpt
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class RecognizerInfer:
    def __init__(self,ckpt_dir,device=None):
        if device is None:
            device = 'cpu'
        self.device = device
        self.model = self.build_model(ckpt_dir)
        self.model.eval()
        self.model.to(self.device)

    def build_model(self,ckpt_dir):
        model = Emotion2Vec_finetume(768,8)
        # model = Emotion_Recognizer_ResNet()
        load_ckpt(model, ckpt_dir, 'model')
        model.to(self.device)
        model.eval()
        return model

    def forward_model(self, inp):
        inp = self.input_to_batch(inp)
        emo_embeds = inp['emo_embeds']
        # emotion = inp['emotion']
        output = self.model(emo_embeds)
        pred = output['logits']
        pred = nn.functional.softmax(pred,dim=1) 
        pred = torch.argmax(pred,dim=1) # 获取每个样本的预测标签
        return pred

    def preprocess_input(self, inp):
        """

        :param inp: {'emo_embed': array}
        :return:
        """
        emo_embed = inp['emo_embed']  
        item = {'emo_embed': emo_embed}
        return item

    def input_to_batch(self, item):
        emo_embeds = torch.Tensor(item['emo_embed'])[None, :].to(self.device)
        batch = {
            'emo_embeds': emo_embeds,
        }
        return batch

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(inp)
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def example_run(cls): 
        inference_pipeline = pipeline(task=Tasks.emotion_recognition,model="iic/emotion2vec_base", model_revision="v2.0.4")
        ### 加载情绪分类模型
        emotion_infer_ins = cls("./emotion2vec_recognizer_seq_10w","cuda:3")      
        emotion_map = {"surprised": 0, "contempt": 1, "sad": 2, "neutral": 3, "disgusted": 4, "happy": 5, "angry": 6, "fear": 7}
        test_items = []
        with open('./libri_test.csv','r') as fp:
            lines = fp.readlines()[1:]
            for l in lines:
                splits = l.strip().split(',')
                item_name = splits[0]
                if item_name[0].isdigit():
                    continue
                emotion = splits[5]
                test_items.append({'item_name':item_name,'emotion':emotion})
        all_wavfps = glob.glob('/home2/xxx/EI_VC/data/raw/LibriTTS/**/*.wav',recursive=True) + \
            glob.glob('/home2/xxx/EI_VC/data/raw/ei_data/**/*.wav',recursive=True)
        item_wavfp_map = {}
        for wavfp in tqdm(all_wavfps):
            item_name = os.path.basename(wavfp).split('.')[0]
            item_wavfp_map[item_name] = wavfp

        print(len(test_items))
        acc = 0.0
        num = 0
        for item in tqdm(test_items):
            item_name,emotion = item['item_name'],item['emotion']
            wav_fp = item_wavfp_map[item_name]
            emo_embed = get_emo_embedding(wav_fp,inference_pipeline)
            # emo_embed = np.load(f'./promptStyle_seq/{item_name}_gen.npy')
            inp ={'emo_embed':emo_embed}
            emotion_pred = emotion_infer_ins.infer_once(inp).item()
            emotion_gt = emotion_map[emotion]
            if emotion_gt==emotion_pred:
                acc += 1
            num += 1
        print('emotion recognizer acc:' ,acc/num)
            
def get_emo_embedding(wavfp,inference_pipeline):
    rec_result = inference_pipeline(wavfp, output_dir="./tmp", granularity="frame")
    emo_embed = rec_result[0]['feats']
    return emo_embed


if __name__ == '__main__':
    RecognizerInfer.example_run()

### 0.41  0.385  0.365
### instruct prompt  style