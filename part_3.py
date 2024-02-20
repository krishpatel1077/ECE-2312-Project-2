import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd

#Sample rate from project 1
sr= 44100

#function to create the chirp signal
def generate_chirp_tone(duration):
    # Time array for the sine tone
    time = np.arange(0, duration, 1/sr)
    # Frequency of the sine tone varying 0 to 8000Hz
    frequency_range = np.linspace(0, 8000, sr*duration)
    # Generate the sine tone
    sine_wave = np.sin(2 * np.pi * frequency * time)
    # Play the sine tone
    sd.play(sine_wave, samplerate=sr)
    sd.wait()  # Wait until the sine tone is done playing
    # Save the sine tone to a WAV file
    write(f'teamKKA-sinetone.wav', sr, sine_wave)

    # Plot the spectrogram of the sine tone
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(sine_wave)), ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Sine Tone')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig('sine_tone_spectrogram.png')  # Save spectrogram as PNG image
    plt.show()