from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import glob
import os
from tqdm import tqdm


'''
Using the emotion representation model
rec_result only contains {'feats'}
	granularity="utterance": {'feats': [*768]}
	granularity="frame": {feats: [T*768]}
'''
inference_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_base", model_revision="v2.0.4")
all_wavs = glob.glob('/home2/xxx/emotion_disc/wav_promptself/**/*.wav',recursive=True)
for wavfp in tqdm(all_wavs):
    rec_result = inference_pipeline(wavfp, output_dir="./wav_promptself_seq", granularity="frame")
