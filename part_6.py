import librosa
import IPython as ip
from scipy.io.wavfile import write
import numpy as np
import sounddevice as sd

y1, sample_rate1 = librosa.load('teamKKA-speechchirp.wav',sr=44100, mono=True)
