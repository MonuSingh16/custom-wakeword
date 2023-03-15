import numpy as np
import pandas as pd

import librosa
import python_speech_features


n_mfccs = 128
len_mfcc= 36

def audio_features(file_path):
     
    x , sr = librosa.load(file_path)
    mfcc = python_speech_features.base.mfcc(x, samplerate=sr, winlen=0.256,
                                        winstep=0.050, numcep=16, nfilt=26,
                                        nfft=8192, preemph=0.0, ceplifter=0,
                                        appendEnergy=False, winfunc=np.hanning).T
    

    # Only keep MFCCs with given length
    if mfcc.shape[1] < len_mfcc:
        pad_width = len_mfcc - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :len_mfcc]
        
    X_inf = np.concatenate(mfcc, axis=0).reshape(1, 16, 36)
    X_inf = X_inf.reshape(X_inf.shape[0], 
                        X_inf.shape[1], 
                        X_inf.shape[2], 
                        1)

    return X_inf