import time
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from model_profiler import model_profiler


from io import StringIO
import streamlit as st
from PIL import Image



from features import audio_features
from record_audio import record_audio

st.title("Wakeword Detection")
image = Image.open('../imgs/display-image.png')
st.image(image, caption='Wakeword Detection')
st.info("""To design, build and deploy a lightweight Keyword spotting ML model
        (CNN, SVM) and exposed as a mobile-application that can process a “custom wake word”.
          Voice response with results by respecting local device resource constraints (low compute) 
          and adhering to ethical challenges (Privacy respecting and non-eavesdropping) """)

app_form = st.sidebar.form(key="Wakeword Detection")

uploaded_file = app_form.file_uploader("Choose a pre-recorded .wav file")
model_select = app_form.selectbox("Choose a Model", ("CNN", "LSTM"))
submit_button = app_form.form_submit_button(" Execute ")



if submit_button:
    X_inf = audio_features(file_path=uploaded_file)
    start = time.perf_counter()
    if model_select == "CNN":
        model_cnn = models.load_model('saved_models/model_cnn.h5')
        Batch_size = 1
        profile = model_profiler(model_cnn, Batch_size)
        st.header('Model Profiler -')
        st.write(profile)
        y_pred = model_cnn.predict(X_inf)
        print_result = ['This is not a wake word', 'This is a wake word']
        res_val = (y_pred>=0.7).astype('int')
        if res_val==1:
            print(print_result[int(res_val)], 'with a probablity score of : ', y_pred[0][0])
            st.write("")
            st.success('Done', icon="✅")
            st.snow()
            st.write("This is a wake word with a probability score of %.3f " %y_pred[0][0])
        else:
            print(print_result[int(res_val)], 'with a probablity score of : ', 1-y_pred[0][0])
            st.write("")
            st.error('Error', icon="🚨")
            st.write("This is not a wake word with a probablity score of %.3f : " %(1-y_pred[0][0]))
    else:
        model_cnn = models.load_model('saved_models/model_cnn.h5')
        y_pred = model_cnn.predict(X_inf)
        print_result = ['This is not a wake word', 'This is a wake word']
        res_val = (y_pred>=0.7).astype('int')
        if res_val==1:
            print(print_result[int(res_val)], 'with a probablity score of : ', y_pred[0][0])
            st.success('Done')
            st.write("This is a wake word ")
        else:
            print(print_result[int(res_val)], 'with a probablity score of : ', 1-y_pred[0][0])
            st.error('This is an error', icon="🚨")
            st.write("This is not a wake word ")
    
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
    st.info('Elapsed time to run through prediction function is %.3f seconds.' % elapsed, icon="ℹ️")
    # The .3f is to round to 3 decimal places.






 