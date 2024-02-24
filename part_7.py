from scipy.io.wavfile import write
import numpy as np
import librosa.display
import sounddevice as sd
import matplotlib.pyplot as plt

sr=44100

# Load the audio file
y, sr = librosa.load(('teamKKA-speechchirp.wav'), sr=44100, mono=True)
y2, sr2 = librosa.load(('quick_brown_fox_MONO.wav'), sr=44100, mono=True)

#create stereo array
stereo_array = np.array([y,y2])
#transpose array to be vertical
stereo_out = np.transpose(stereo_array)

# Play the stereo tone
sd.play(stereo_out, samplerate=sr)
sd.wait()  # Wait until the stereo tone is done playing

#write stereo audio to file
write("teamKKA-stereospeechsine.wav", sr, stereo_out)


# Plot the two spectrograms in subplots
fig, (ax1, ax2) = plt.subplots(2, sharey=True)

#Create spectrogram for the speechchirp
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max), sr=sr, x_axis='time', y_axis='log', ax=ax1)
#Turn off specshow axis lables
plt.ylabel(None)
plt.xlabel(None)
ax1.set_title('Speech With Sine Wave')

#Create spectrogram for voice only
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max), sr=sr, x_axis='time', y_axis='log', ax=ax2)
#Turn off specshow axis lables
plt.ylabel(None)
plt.xlabel(None)
ax2.set_title('Quick Brown Fox Recording')
#Label both x and y axis for entire plot
fig.supxlabel('Time (s)')
fig.supylabel('Frequency (Hz)')


plt.tight_layout()
plt.savefig('stereospeechsine_spectrogram.png')  # Save spectrogram as PNG image

plt.show()