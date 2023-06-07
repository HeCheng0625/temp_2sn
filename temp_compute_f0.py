import librosa
import pyworld as pw
import numpy as np
import os

# wav_dir = "/blob/v-yuancwang/11lab_ns2_infer_steps/GeneralSentence"
wav_dir = "/blob/v-yuancwang/LATENT_TTS/ft_11lab_frame_level/generated_0__v1_tp1.2"
# wav_dir = "/blob/v-yuancwang/TTS_Data/zeqian_test/frame_level"

f0_means, f0_stds, f0_ranges, f0_coarse_means = [], [], [], []
for wav_path in os.listdir(wav_dir):
    wav_path = os.path.join(wav_dir, wav_path)
    wav, sr = librosa.load(wav_path, sr=16000)
    

    # pyworld
    wav = wav.astype(np.double)
    f0, t = pw.dio(wav, sr, frame_period = 200 / sr * 1000)
    f0 = pw.stonemask(wav, f0, t, sr)
    
    f0 = f0[f0 != 0.]
    f0_means.append(f0.mean())
    f0_stds.append(f0.std())
    f0_ranges.append(f0.max()-f0.min())

print("f0_mean_mean:", sum(f0_means)/len(f0_means))
print("f0_std_mean:", sum(f0_stds)/len(f0_stds))
print("f0_range_mean:", sum(f0_ranges)/len(f0_ranges))
