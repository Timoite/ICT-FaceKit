import os
from wavlm_lora import WavLMWrapper
import torch
import torchaudio
import scipy.signal as sig
from tqdm import tqdm
import numpy as np

#8Hz low-pass filter
b,a = sig.butter(5,8,"low",fs=50)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Instantiate wrapper and load model
inv_model = WavLMWrapper().to(device)
state_dict = torch.load("inversion_checkpoints/lora_multispeaker_consistency_alpha_0.25_threshold_0_vctk_vvn_4tonguepoints")
inv_model.load_state_dict(state_dict)
inv_model.eval()

#data,sr = torchaudio.load("./inputs/test.wav")
#
#with torch.no_grad():
#    ema = inv_model(data.to(device)).squeeze().detach().cpu().numpy()
#
#ema = sig.filtfilt(b,a,ema,axis=0)
#
#np.save("outputs/motion.npy",ema)

base_dir = "./26/"

files = os.listdir(os.path.join(base_dir,"wav"))

for f in tqdm(files):
    wav,sr = torchaudio.load(os.path.join(base_dir,"wav",f))

    split_len = 160000
    splits = wav.shape[1]//split_len

    for i in range(splits):
        wav_split = wav[:,i*split_len:(i+1)*split_len]

        torchaudio.save(os.path.join(base_dir,"wav_split",f[:-4]+f"_{i}"+".wav"),wav_split,sr)

        #Inversion
        with torch.no_grad():
            ema = inv_model(wav_split.to(device)).squeeze().detach().cpu().numpy()

        #Low-pass filter
        ema = sig.filtfilt(b,a,ema,axis=0)

        np.save(os.path.join(base_dir,"npy_split",f[:-4]+"_{i}"+".npy"),ema)
