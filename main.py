import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
from keras.models import load_model
from keras import backend as K
import pyaudio
import webbrowser
import wave 
import keras

def predictSound(X):
    y, sr = librosa.load(X)
    noisy_part = y[8000:]
    y = nr.reduce_noise(audio_clip=y, noise_clip=noisy_part, verbose=False)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect = np.expand_dims(spect, axis = -1)
    spect = np.expand_dims(spect, axis = 0)
    model1 = load_model()
    res = model1.predict(np.array(spect))
    return res
def load_model():
    loaded = keras.models.load_model('./updated_model.h5')
    return loaded
if __name__ == "__main__":
    st.title("Alpha Ai Solution")
    st.subheader("Cough Detection Web Application")
    activites = ["Members","Chirag","Garima","Arshid","Ritik"]
    choice = st.sidebar.selectbox("Alpha Team",activites)
    status = st.radio("Activate the App",("Start","Stop"))
    if status == "Start" :
        st.success("its Activated")
        while(True):
            FORMAT = pyaudio.paFloat32
            CHANNELS = 2
            RATE = 22050
            CHUNK = 1024
            RECORD_SECONDS = 4
            WAVE_OUTPUT_FILENAME = "./file.wav"
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
            if res[0][1] >= 0.5 :
                #expected output to give condition
                webbrowser.open('http://coughdetect.c1.biz/', new=2)
                break
            stream.stop_stream()
            stream.close()
            p.terminate()
        # voice_recording()
    if status == "Stop":
        st.error("Stoped")
