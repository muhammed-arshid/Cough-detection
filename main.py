import streamlit as st
import joblib,os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import noisereduce as nr
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import IPython
import os
import pyaudio
import webbrowser 

def load_model():
    model_name = "/home/arshid/Desktop/pro/updated_model"

    # Model reconstruction from JSON file
    with open( model_name + '.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights( model_name + '.h5')
    return model


def predictSound(X):
    y, sr= librosa.load(x)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    model1 = load_model()
    res = model1.predict(np.array(spect))
    return res

def main():
    st.title('Alpha Ai Solution')
    st.subheader('Cough Detection Web Application')
    activites = ["Members","mem 1","mem 1","mem 1","mem 1"]
    choice = st.sidebar.selectbox("Alpha Team",activites)
    status = st.radio("Activate the App",("Start","Stop"))
    if status == "Start" :
        st.success("its Activated")
        CHUNKSIZE = 22050 # fixed chunk size
        RATE = 22050

        # initialize portaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

        #noise window
        data = stream.read(4000)
        # noise_sample = np.frombuffer(data, dtype=np.float32)
        # loud_threshold = np.mean(np.abs(noise_sample)) * 10
        audio_buffer = []
        near = 0

        while(True):
            # Read chunk and load it into numpy array.
            data = stream.read(CHUNKSIZE)
            current_window = np.frombuffer(data, dtype=np.float32)
            noise_sample = np.frombuffer(data, dtype=np.float32)
            #Reduce noise real-time
            current_window = nr.reduce_noise(audio_clip=current_window, noise_clip=noise_sample, verbose=False)
            
            if(audio_buffer==[]):
                audio_buffer = current_window
            else:
                    if(near<4):
                        audio_buffer = np.concatenate((audio_buffer,current_window))
                        near += 1
                    else:
                        res  = predictSound(np.array(audio_buffer))
                        st.write(res)
                        if res == 1 :#expected output to give condition
                            webbrowser.open('http://coughdetect.c1.biz/', new=2) #create a external page and frre host pass that url into heare
                        audio_buffer = []
                        near

        # close stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # voice_recording()
    if status == "Stop":
        st.error("Stoped")
if __name__ == "__main__":
    main()
