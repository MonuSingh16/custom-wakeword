import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tensorflow.keras import models
from model_profiler import model_profiler

import streamlit as st
from PIL import Image
from features import audio_features

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Wakeword Detection")
image = Image.open('../spotify-custom-wakeword/imgs/display-image.png')
st.image(image, caption='Wakeword Detection')
with st.expander("Project Objective"):
    st.write("""To design, build and deploy a lightweight Keyword spotting ML model
        (CNN, SVM) and exposed as a mobile-application that can process a “custom wake word”.
          Voice response with results by respecting local device resource constraints (low compute) 
          and adhering to ethical challenges (Privacy respecting and non-eavesdropping) """)

app_form = st.sidebar.form(key="Wakeword")
app_form.image("https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png" )    
uploaded_file = app_form.file_uploader("Choose a pre-recorded .wav file")

# Add model select boxes
model_select = app_form.selectbox(
        "Choose Model:",
        ("CNN", "ResNet", "LSTM")
    )
submit_button = app_form.form_submit_button(" Execute ")

if submit_button:
    x, sr, X_inf = audio_features(file_path=uploaded_file)
    start = time.perf_counter()
    if model_select == "CNN":
        model_cnn = models.load_model('../spotify-custom-wakeword/app/saved_models/model_cnn.h5')
        
        tab1, tab2, tab3, tab4 = st.tabs(["🗃 Raw Data", "✅ Model Results", "🔎 Model Details", "🤓 Model Evaluation"])
        
        with tab1:
            st.header("Raw Data")
            plt.subplot(211)
            plt.title('Spectrogram of a wav file')
            plt.plot(x)
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            st.pyplot()

            plt.subplot(212)
            plt.specgram(x,Fs=sr)
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            st.pyplot()
                    
        with tab2:

            st.header("Model Results")
            y_pred = model_cnn.predict(X_inf)
            print_result = ['This is not a wake word', 'This is a wake word']
            y_pred_int = np.argmax(y_pred, axis=1)            
            if y_pred_int==1:
                print(print_result[int(y_pred_int)], 'with a probablity of : ', y_pred[0][y_pred_int][0])
                st.write("")
                
                audio_file = open('../spotify-custom-wakeword/app/speech-wakeword.wav', 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/ogg')
                st.snow()
                st.success('This is a wake word', icon="✅")
                st.info("""The selected model results this selected file as a wake word \n
                         Accuracy score of %.3f""" %y_pred[0][y_pred_int][0]
                        ,icon="ℹ️")
            else:
                print(print_result[int(y_pred_int)], 'with a probablity of : ', y_pred[0][y_pred_int][0])
                st.write("")
                audio_file = open('../spotify-custom-wakeword/app/speech-otherword.wav', 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/ogg')
                st.error('This is not a wake word', icon="🚨")
                st.info("""The selected model results this selected file as not a wake word \n 
                            Accuracy score of %.3f""" %y_pred[0][y_pred_int][0]
                        ,icon="ℹ️")
        
        with tab3:    
            st.header("Model Architecture")
            st.image('../spotify-custom-wakeword/imgs/cnn-model.png', caption='CNN Model', use_column_width='always', clamp=True)
        
            Batch_size = 128
            profile = model_profiler(model_cnn, Batch_size)
            st.header('Model Profiler')
            st.write(profile)
        
        with tab4:
            st.header("Model Evaluation")
            model_results_df = pd.read_csv("../spotify-custom-wakeword/app/results/model_results.csv", index_col=False)
            st.write(model_results_df)
            st.image('../spotify-custom-wakeword/imgs/confusion-matrix.png')
        
    else:
        st.write("No Model Selected")

    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
    st.info('Elapsed time to run through prediction function is %.3f seconds.' % elapsed, icon="ℹ️")
    # The .3f is to round to 3 decimal places.



 