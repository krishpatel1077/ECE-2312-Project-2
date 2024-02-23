import librosa
import IPython as ip
from scipy.io.wavfile import write

#samplind_rate1 = sampling_rate2 = 44100
y1, sample_rate1 = librosa.load('quick_brown_fox_MONO.wav', mono=True)
y2, sample_rate2 = librosa.load('teamKKA-sinetone.wav', mono=True)

print(librosa.util.stack([y1, y2], axis=1))


#write(f"teamKKA-speechchirp.wav", sr, )