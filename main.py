import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
from keras.models import load_model
import pyaudio
import webbrowser
import wave 
import keras
def svaeWavefile(fileName):
    waveFile = wave.open(fileName, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(p.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def preeProcessing(X,model):
    y, sr = librosa.load(X)
    noisy_part = y[7000:]
    y = nr.reduce_noise(audio_clip=y, noise_clip=noisy_part, verbose=False)
    spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
    spect = librosa.power_to_db(spect, ref=np.max)
    spect = np.expand_dims(spect, axis = (-1,0,))
    res = model.predict(np.array(spect))
    return res
if __name__ == "__main__":
    st.title("Alpha Ai Solution")
    st.subheader("Cough Detection Web Application")
    activites = ["Members","Chirag","Garima","Arshid","Ritik"]
    choice = st.sidebar.selectbox("Alpha Team",activites)
    status = st.radio("Activate the App",("Start","Stop"))
    model = keras.models.load_model('./final_model2.h5')
    if status == "Start" :
        st.success("its Activated")
        while(True):
            FORMAT = pyaudio.paFloat32
            CHANNELS = 2
            RATE = 44100
            CHUNK = 1024
            RECORD_SECONDS = 4
            WAVE_OUTPUT_FILENAME = "./file.wav"
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=2, rate=RATE, input=True, frames_per_buffer=CHUNK)
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)
            svaeWavefile(WAVE_OUTPUT_FILENAME)
            file1 = './file.wav'
            res  = preeProcessing(file1,model)
            print(res)
            if res[0][1] >= 0.5 :
                #expected output to give condition
                webbrowser.open('http://coughdetect.c1.biz/', new=2)
            stream.stop_stream()
            stream.close()
            p.terminate()
        # voice_recording()
    if status == "Stop":
        st.error("Do you want to start? \n just press ''START''")
