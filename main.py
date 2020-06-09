import streamlit as st
import joblib,os
import numpy as np
import librosa
import noisereduce as nr
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import pyaudio
import webbrowser
import wave 

def predictSound(X):
    y, sr = librosa.load(X)
    noisy_part = y[8000:]
    y = nr.reduce_noise(audio_clip=y, noise_clip=noisy_part, verbose=False)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect = np.expand_dims(spect, axis = -1)
    spect = np.expand_dims(spect, axis = 0)
    return np.array(spect)
def model1():
    model_weights_path = "./weights1.h5"
    model_path = "./model.h5"
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    return model
if __name__ == "__main__":
    st.title("Alpha Ai Solution")
    st.subheader("Cough Detection Web Application")
    activites = ["Members","mem 1","mem 1","mem 1","mem 1"]
    choice = st.sidebar.selectbox("Alpha Team",activites)
    status = st.radio("Activate the App",("Start","Stop"))
    if status == "Start" :
        st.success("its Activated")
        t = True
        while(t):
            FORMAT = pyaudio.paFloat32
            CHANNELS = 2
            RATE = 22050
            CHUNK = 1024
            RECORD_SECONDS = 4
            WAVE_OUTPUT_FILENAME = "file.wav"
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=2, rate=RATE, input=True, frames_per_buffer=CHUNK)
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(p.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
            file1 = './file.wav'
            res  = predictSound(file1)
            model =model1()
            predictions = np.round(model.predict(res))
            print(predictions)
            if predictions[0][0] == 1 :#expected output to give condition
                webbrowser.open('http://coughdetect.c1.biz/', new=2)
                t = False
            stream.stop_stream()
            stream.close()
            p.terminate()
        # voice_recording()
    if status == "Stop":
        st.error("Stoped")
