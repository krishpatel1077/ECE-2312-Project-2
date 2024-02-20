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
    time = np.linspace(0, duration, int(sr*duration))
    # Frequency of the sine tone varying 0 to 8000Hz
    frequency = np.linspace(0, 8000, len(time))
    
    output = frequency*time
    print("this is the output: \n",output)

    # Generate the sine tone
    chirp_wave = np.sin(2 * np.pi * frequency * time)
    # Play the sine tone
    sd.play(chirp_wave, samplerate=sr)
    sd.wait()  # Wait until the sine tone is done playing
    # Save the sine tone to a WAV file
    write(f'teamKKA-chirp.wav', sr, chirp_wave)

    # Plot the spectrogram of the sine tone
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(chirp_wave)), ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Chirp Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig('chirp_spectrogram.png')  # Save spectrogram as PNG image
    plt.show()
    
# Ask for speech duration input (I'm setting this to 7 seconds by default)
speech_duration = float(input("Enter the duration of the speech file in seconds: "))

# Generate sine tone with the provided duration
generate_chirp_tone(speech_duration)
