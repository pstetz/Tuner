import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import numpy as np
import pyaudio
import time
import wave
import sys

import warnings
warnings.filterwarnings("ignore")

def record_wav(filepath, CHUNK=1024, FORMAT=pyaudio.paInt16, CHANNELS=1, RATE=44100):
    p = pyaudio.PyAudio()
    stream = p.open(
        format = FORMAT,
        channels = CHANNELS,
        rate = RATE,
        input = True,
        frames_per_buffer = CHUNK
    )

    frames = []
    for i in range(0, int(RATE / CHUNK * 2)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filepath, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def plot_wav(filepath):
    spf = wave.open(filepath,'r')

    #Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')


    #If Stereo
    if spf.getnchannels() == 2:
        sys.exit(0)

    plt.figure(1)
    plt.title('Signal Wave...')
    plt.plot(signal)
    plt.show()

### Configure
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
filename = "temp.wav"

### Main
while True:
    record_wav(filename)
    plot_wav(filename)
