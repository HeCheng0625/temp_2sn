import torch
import librosa
from usr_dir.codec.codec_encoder import CodecEncoder
from usr_dir.codec.codec_decoder import CodecDecoder
import soundfile as sf
import numpy as np
codec_enc = CodecEncoder()
codec_dec = CodecDecoder()
device = torch.device('cuda:0')
ckpt_path = '/blob/v-shenkai/checkpoints/tts/codec/chanpin_5w/v5/lambda_disc_1_commit_weight_0.25/checkpoint-440000.pt'
checkpoint = torch.load(ckpt_path, map_location='cpu')
codec_enc.load_state_dict(checkpoint["model"]['CodecEnc'])
codec_enc = codec_enc.eval().to(device)
codec_dec.load_state_dict(checkpoint["model"]['generator'])
codec_dec = codec_dec.eval().to(device)
for i in range(1,6):
    wav_fn = f'music/01000000100{i}.wav'
    wav_data, sr = librosa.load(wav_fn, sr=16000)
    wav_data = np.pad(wav_data, (0, 200-len(wav_data) % 200))
    wav_tensor = torch.from_numpy(wav_data).to(device)  
    vq_emb = codec_enc(wav_tensor[None, None, :])
    ref_mel, _, _ = codec_dec(vq_emb, None, 1000000, vq=True)
    ref_audio, _ = codec_dec.inference(ref_mel)
    ref_audio = ref_audio.detach().cpu()
    print(ref_audio.shape)
    sf.write(f'music/rec_01000000100{i}.wav', ref_audio[0][0], 16000, subtype='PCM_24')