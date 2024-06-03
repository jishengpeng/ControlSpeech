from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
# from evaluate import load
import librosa
import pandas as pd
import tqdm
from datasets import load_metric


# download whisper-large-v3 
def predict(audio, text):
    y, sr = librosa.load(audio, sr=16000)
    input_features = processor(y, sampling_rate=sr, return_tensors="pt", language="english").input_features
    reference = processor.tokenizer._normalize(text)
    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cpu"), language="english")[0]
    transcription = processor.decode(predicted_ids, language="english")
    prediction = processor.tokenizer._normalize(transcription)
    return reference, prediction

processor = WhisperProcessor.from_pretrained("/home/disk2/xxx/Data/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("/home/disk2/xxx/Data/whisper-large-v3").to("cpu")


reference, prediction = [], []
df = pd.read_csv("/home/disk2/xxx/Data/2024nips/0509/libri_test_undomain_timbre_addid.csv")
audio = df['new_id'].tolist()
text = df['txt'].tolist()

for a, t in tqdm.tqdm(zip(audio, text)):
    ref, pred = predict(f"/home/disk2/xxx/Result/vccm/infer/test_gt_codec_timbre/{a}.wav", t)
    reference.append(ref)
    prediction.append(pred)

res = ({
    "reference": reference,
    "prediction": prediction
})
pd.DataFrame(res).to_csv("/home/disk2/xxx/Data/2024nips/0509/jialong/whisper_largev3_codec_result_timbre.csv", index=False)

# input_path = "/home/disk2/xxx/Data/2024nips/0509/jialong/whisper_largev3_codec_result.csv"
# df = pd.read_csv(input_path)
# reference = df["reference"].values
# prediction = df["prediction"].values


wer = load_metric("wer")
print(100 * wer.compute(references=reference, predictions=prediction))
# print(100 * wer.compute(references=reference[200:], predictions=prediction[200:]))