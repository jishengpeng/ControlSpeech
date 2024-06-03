import numpy as np
import glob
import os
from tqdm import tqdm


all_wavs = glob.glob('/home2/xxx/emotion_disc/promptStyle/**/gen.wav',recursive=True)
for wavfp in tqdm(all_wavs):
    item_name = os.path.basename(os.path.dirname(wavfp))
    os.rename(wavfp,os.path.join(os.path.dirname(wavfp),f'{item_name}_gen.wav'))