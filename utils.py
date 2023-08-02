import librosa
import librosa.display
from librosa import feature
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def extract_mfccs(relative_path: str = 'an4/test/an4/wav/an391-mjwl-b.wav') -> List[np.array]:

    """load wav file,
    # split into t units
    #   apply stft with hanning window (fft on parts of the data, while windowing it)
    #   apply mfcc
    #   add to res list
    return mfcc-s
    """

    # Replace 'your_audio_file_path' with the actual relative_path to your audio file
    y, sr = librosa.load(relative_path)
    res = []

    # Apply the Hann window to the audio signal
    win_length = 2205  # Same as n_fft for full coverage
    hop_length = 512
    ys = np.array_split(np.array(y), 20)
    ys = ys * np.hanning(win_length)

    for ys_i in ys:
        # Calculate the Short-Time Fourier Transform (STFT)
        D = np.abs(librosa.stft(ys_i, n_fft=win_length, hop_length=hop_length))

        # Extract the Mel-frequency cepstral coefficients (MFCCs)
        n_mfcc = 16
        mfcc = feature.mfcc(S=librosa.amplitude_to_db(D), n_mfcc=n_mfcc)
        res.append(mfcc)

    t = np.hstack(res)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(t, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

    return res