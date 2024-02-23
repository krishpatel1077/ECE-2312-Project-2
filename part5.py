import librosa
import IPython as ip
from scipy.io.wavfile import write
import numpy as np
import sounddevice as sd

y1, sample_rate1 = librosa.load('quick_brown_fox_MONO.wav', mono=True)
y2, sample_rate2 = librosa.load('teamKKA-sinetone.wav', mono=True)
sr=44100

output = y1+(y2*0.5)

sd.play(output, samplerate=sr)
sd.wait()
write(f"teamKKA-speechchirp.wav", sr, output)