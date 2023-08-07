import librosa
import librosa.display
from librosa import feature
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List
import os

from torch.utils.data import Dataset
import string


def create_index(additional_letters: str):
    alphabet = '<BLANK>' + ' ' + '<@>' + string.ascii_lowercase() + additional_letters
    index_letter_pairing = dict()
    for i, l in enumerate(alphabet):
        index_letter_pairing[i] = l
    return index_letter_pairing


def extract_mfcc(audio_data: torch.Tensor) -> torch.Tensor:
    """
    :param audio_data: (torch.Tensor): Input audio data of shape (n, time)
    :return: MFCC features of shape (n, n_mfcc, t)
    """
    mfcc_features = []

    for audio in audio_data:
        mfcc = librosa.feature.mfcc(y=audio.numpy())
        mfcc_features.append(torch.tensor(mfcc))

    mfcc_features = torch.stack(mfcc_features)
    return mfcc_features


# get CTC loss
def calculate_probability(matrix_path: torch.Tensor, labels: str, alphabet: list):
    """
    :param matrix_path: A relative_path to a 2D numpy matrix of network outputs (y)
    :param labels: A string of the labeling you wish to calculate the probability for
    :param alphabet: A string specifying the possible output tokens
    :return: The computed probability calculated with CTC
    """

    probabilities_matrix = matrix_path

    # inserting the empty character, implemented as @
    labels = '@' + '@'.join([char for char in labels]) + '@'
    ctc_matrix = np.zeros([len(labels), len(probabilities_matrix)])  # create a matrix of shape [T, K]

    # initialize
    ctc_matrix[0, 0] = probabilities_matrix[0, 0]  # alpha_1,1 = y^1_e
    ctc_matrix[1, 0] = probabilities_matrix[0, alphabet.index(labels[1])]  # alpha_2,1 = y^1_z1

    # calculate ctc
    for i in range(ctc_matrix.shape[0]):
        char_index = alphabet.index(labels[i])
        for j in range(1, ctc_matrix.shape[1]):
            tmp_sum = ctc_matrix[i - 1, j - 1] + ctc_matrix[i, j - 1]
            if labels[i] == '@' or labels[i] == labels[i - 2]:  # z_s = epsilon or z_s = z_s-2
                ctc_matrix[i, j] = tmp_sum * probabilities_matrix[j][char_index]
            else:
                ctc_matrix[i, j] = (tmp_sum + ctc_matrix[i - 2, j - 1]) * probabilities_matrix[j][char_index]

    return ctc_matrix[-1, -1] + ctc_matrix[-2, -1]


def load_wav_files(paths):
    """
    :param paths: either a path to directory of .wav files or a list of paths to files
    :return: tensor of shape [number of files, samples per file], samples per file is the data of each file
    """
    spectogram_list = []

    if isinstance(paths, str):
        # If a single directory path is provided
        if os.path.isdir(paths):
            file_list = [os.path.join(paths, file_name) for file_name in os.listdir(paths) if
                         file_name.endswith(".wav")]
        else:
            raise ValueError("Invalid directory path:", paths)
    elif isinstance(paths, list):
        # If a list of file paths is provided
        file_list = [path for path in paths if os.path.isfile(path) and path.endswith(".wav")]
    else:
        raise ValueError("Invalid paths argument:", paths)

    max_len = 0
    input_len_list = []
    for file_path in file_list:
        waveform, sr = librosa.load(file_path, mono=True)
        spec = librosa.feature.melspectrogram(y=waveform, sr=sr).T  # extract (128,T) convert to (T,128)
        spectogram_list.append(spec)
        input_len_list.append(spec.shape[0])
        if max_len < spec.shape[0]:
            max_len = spec.shape[0]

    for (i, spec) in enumerate(spectogram_list):
        if spec.shape[0] < max_len:
            pad_len = max_len - spec.shape[0]
            spectogram_list[i] = np.pad(spec, ((0, pad_len), (0, 0)), mode='constant')

    spectrogram_tensor = torch.stack([torch.from_numpy(spec) for spec in spectogram_list])
    input_len_tensor = torch.tensor(data=input_len_list)
    return spectrogram_tensor, input_len_tensor


def get_file_in_dir(path):
    """
    :param path: path to a directory
    :return: list of the names of files in the directory
    """
    file_names = []

    for file_name in os.listdir(path):
        file_names.append(os.path.join(path, file_name))

    return file_names


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
