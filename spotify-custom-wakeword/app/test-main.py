import time

import matplotlib.pyplot as plt
from tensorflow.keras import models
from model_profiler import model_profiler


import streamlit as st
from PIL import Image

from features import audio_features
import librosa

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Wakeword Detection")
image = Image.open('../imgs/display-image.png')
st.image(image, caption='Wakeword Detection')
st.info("""To design, build and deploy a lightweight Keyword spotting ML model
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
        model_cnn = models.load_model('saved_models/model_cnn.h5')
        
        tab1, tab2, tab3, tab4 = st.tabs(["🗃 Raw Data", "✅ Model Results", "🔎 Model Details", "🤓 Model Explainability"])
        
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
        
        with tab3:    
            st.header("Model Details")
            st.image('../imgs/cnn-model.png', caption='CNN Model', use_column_width='always', clamp=True)
        
            Batch_size = 1
            profile = model_profiler(model_cnn, Batch_size)
            st.header('Model Profiler -')
            st.write(profile)
        
        with tab4:
            st.header("Model Explainability")
        
    else:
        st.write("No Model Selected")

    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
    st.info('Elapsed time to run through prediction function is %.3f seconds.' % elapsed, icon="ℹ️")
    # The .3f is to round to 3 decimal places.



 