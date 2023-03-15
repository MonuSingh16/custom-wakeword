import pandas as pd
import numpy as np

from tensorflow.keras import layers, models
from features import audio_features
from record_audio import record_audio

import streamlit as st

st.title('Custom Wake-Word')

st.button("Record")

wav_path = '../dat/other_words/33.wav'
X_inf = audio_features(file_path=wav_path)

model_cnn = models.load_model('saved_models/model_cnn.h5')
y_pred = model_cnn.predict(X_inf)

print_result = ['This is not a wake word', 'This is a wake word']
res_val = (y_pred>=0.7).astype('int')
if res_val==1:
    print(print_result[int(res_val)], 'with a probablity score of : ', y_pred[0][0])
else:
    print(print_result[int(res_val)], 'with a probablity score of : ', 1-y_pred[0][0])