import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd

# Define Sampling Rate or Frequency in Hz (taken from quick brown fox part 1)
sr = 44100

def generate_sine_tone(duration):
    # Time array for the sine tone
    time = np.arange(0, duration, 1/sr)
    # Frequency of the sine tone
    frequency = 5000
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

# Ask for speech duration input (I'm setting this to 7 seconds by default)
speech_duration = float(input("Enter the duration of the speech file in seconds: "))

# Generate sine tone with the provided duration
generate_sine_tone(speech_duration)
