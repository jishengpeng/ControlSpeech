import os
import glob
import librosa
import torch
from tqdm import tqdm

from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import warnings
warnings.filterwarnings('ignore')


### “wavlm-base-plus-sv” dowanload link：https://huggingface.co/microsoft/wavlm-base-plus-sv

# load the pre-trained checkpoints
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('/home/disk2/xxx/Data/wavlm-base-plus-sv')
model = WavLMForXVector.from_pretrained('/home/disk2/xxx/Data/wavlm-base-plus-sv').to(device='cuda:0')
model.eval()

def get_wavlm_embed(audios):
    with torch.no_grad(): ###here
        inputs = feature_extractor(audios, padding=True, return_tensors="pt",sampling_rate=16000).to(device='cuda:0')
        output = model(**inputs)
        embed = output.embeddings
        return embed

import logging

training_log_path="/home/disk2/xxx/Result/vccm/infer/metrics/spk_wavlm_sv/controlspeech_small_bert_mdn001_3_lr0005_warm4000_epoch3_undomainstyle.log"

os.system("rm %s"%(training_log_path))


LOG_FORMAT = "time: %(asctime)s - level: %(levelname)s - information: %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, filename=training_log_path)

path="/home/disk2/xxx/Result/controlspeech/infer/controlspeech_small_bert_mdn001_3_lr0005_warm4000_epoch3_undomainstyle_samples/wav_promptself"
ref_path="/home/disk2/xxx/Result/vccm/infer/test_gt_undomainstyle"

cosine_sim = torch.nn.CosineSimilarity(dim=-1)


sim_scores = []

num = 0

for i in os.listdir(path):
    # breakpoint()
    if(os.path.join(path,i).endswith(".wav")):
        wav1_path=os.path.join(path,i)
        wav2_path=os.path.join(ref_path,i)

        y1,_ = librosa.load(wav1_path,sr=16000)
        y2,_ = librosa.load(wav2_path,sr=16000)

        audio1 = []
        audio2 = []
        audio1.append(y1)
        audio2.append(y2)
    try:
        embed1 = get_wavlm_embed(audio1)
        embed2 = get_wavlm_embed(audio2)
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        continue
    embed1 = torch.nn.functional.normalize(embed1, dim=-1).cpu()
    embed2 = torch.nn.functional.normalize(embed2, dim=-1).cpu()
    for j in range(embed2.shape[0]):
        similarity = cosine_sim(embed1[j], embed2[j]).item()
        sim_scores.append(similarity)
        num += 1
        # breakpoint()
        logging.info(i+"|"+str(similarity))

logging.info("final")
logging.info(sum(sim_scores)/num)
        
        