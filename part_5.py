import librosa
import IPython as ip
from scipy.io.wavfile import write
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

y1, sample_rate1 = librosa.load('quick_brown_fox_MONO.wav',sr=44100, mono=True)
y2, sample_rate2 = librosa.load('teamKKA-sinetone.wav', sr=44100,mono=True)
sr=44100

output = y1+(y2*0.5)

sd.play(output, samplerate=sr)
sd.wait()
write(f"teamKKA-speechchirp.wav", sr, output)

# Plot the spectrogram of the sine tone
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(output)), ref=np.max), sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of Speechchirp Signal')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('speechchirp_spectrogram.png')  # Save spectrogram as PNG image
plt.show()