import streamlit as st
import joblib,os
import sounddevice
import scipy
from scipy.io.wavfile import write

def voice_recording():
    fs = 44100
    sec = 10
    print("recording...")
    record_voice = sounddevice.rec(int(sec *fs),samplerate=fs,channels=2)
    sounddevice.wait()
    a = scipy.io.wavfile.write("out.wav",fs,record_voice)
    print("finished.........")

def main():
    st.title('Alpha Ai Solution')
    st.subheader('Cough Detection Web Application')
    activites = ["Members","mem 1","mem 1","mem 1","mem 1"]
    choice = st.sidebar.selectbox("Alpha Team",activites)

    status = st.radio("Activate the App",("Start","Stop"))
    if status == "Start" :
        st.success("its Activated")
        voice_recording()
    if status == "Stop":
        st.error("Stoped the fuction ")
