import string

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa
from jiwer import wer

alphabet = list(string.ascii_lowercase + ' ' + '@')  # gives us the a-z, spacebar and @ for epsilon


# get CTC loss
def calculate_probability(matrix_path: torch.Tensor, labels: str):
    """
    :param matrix_path: A path to a 2D numpy matrix of network outputs (y)
    :param labels: A string of the labeling you wish to calculate the probability for
    :use alphabet: A string specifying the possible output tokens
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


def build_probability_matrix():
    pass


class NeuralNetwork(nn.Module):
    def __init__(self, conv_kernels: list, n_labels: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv_kernels = conv_kernels
        self.conv0 = nn.Conv1d(1, conv_kernels[0], 3, padding=1)
        self.conv1 = nn.Conv1d(conv_kernels[1 - 1], conv_kernels[1], 3, padding=1)
        self.conv2 = nn.Conv1d(conv_kernels[2 - 1], conv_kernels[2], 3, padding=1)
        self.conv3 = nn.Conv1d(conv_kernels[3 - 1], conv_kernels[3], 3, padding=1)
        self.conv4 = nn.Conv1d(conv_kernels[4 - 1], conv_kernels[4], 3, padding=1)
        self.conv5 = nn.Conv1d(conv_kernels[5 - 1], conv_kernels[5], 3, padding=1)
        self.conv6 = nn.Conv1d(conv_kernels[6 - 1], conv_kernels[6], 3, padding=1)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool1d(3)
        self.linear = nn.Linear(64 * conv_kernels[-1], n_labels)

        # a more pythonic way to go about this:
        # self.convs = [nn.Conv1d(1, conv_kernels[0], 3, padding=1)] + \
        #              [nn.Conv1d(conv_kernels[i - 1], conv_kernels[i], 3, padding=1) for i in
        #               range(1, len(conv_kernels))]

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.conv0(x))  # 1024
        x = self.relu(self.conv1(x))  # 1024
        x = self.maxpooling(self.relu(self.conv2(x)))  # 512
        x = self.maxpooling(self.relu(self.conv3(x)))  # 256
        x = self.maxpooling(self.relu(self.conv4(x)))  # 128
        x = self.maxpooling(self.relu(self.conv5(x)))  # 64
        x = self.relu(self.conv6(self.flatten(x)))  # 64 x conv_kernels[6 - 1]
        x = nn.Softmax(self.linear(x), dim=1)

        return x

# build NW - so i was thinking to extract "mfcc features" and to train on those features.

# train NW


print("Hellow Neriya!")
