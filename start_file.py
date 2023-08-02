import string
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import tp as tp
from jiwer import wer
import utils

alphabet = list(string.ascii_lowercase + ' ' + '@')  # gives us the a-z, spacebar and @ for epsilon
train_path = r'an4\\train\\an4\\'


class CharacterDetectionNet(nn.Module):
    def __init__(self, classifierArgs):
        super().__init__()
        self.flatten = nn.Flatten()
        conv_kernels = classifierArgs.kernels_per_layer
        self.conv0 = nn.Conv1d(1, conv_kernels[0], 3, padding=1)
        self.conv1 = nn.Conv1d(conv_kernels[1 - 1], conv_kernels[1], 3, padding=1)
        self.conv2 = nn.Conv1d(conv_kernels[2 - 1], conv_kernels[2], 3, padding=1)
        self.conv3 = nn.Conv1d(conv_kernels[3 - 1], conv_kernels[3], 3, padding=1)
        self.conv4 = nn.Conv1d(conv_kernels[4 - 1], conv_kernels[4], 3, padding=1)
        self.conv5 = nn.Conv1d(conv_kernels[5 - 1], conv_kernels[5], 3, padding=1)
        self.conv6 = nn.Conv1d(conv_kernels[6 - 1], conv_kernels[6], 3, padding=1)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool1d(3)
        self.linear = nn.Linear(64 * conv_kernels[-1], len(alphabet))

        # a more pythonic way to go about this:
        # self.convs = [nn.Conv1d(1, conv_kernels[0], 3, padding=1)] + \
        #              [nn.Conv1d(conv_kernels[i - 1], conv_kernels[i], 3, padding=1) for i in
        #               range(1, len(conv_kernels))]

    def forward(self, x):
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.maxpooling(self.relu(self.conv2(x)))
        x = self.maxpooling(self.relu(self.conv3(x)))
        x = self.maxpooling(self.relu(self.conv4(x)))
        x = self.maxpooling(self.relu(self.conv5(x)))
        x = self.relu(self.conv6(self.flatten(x)))
        x = nn.Softmax(self.linear(x), dim=1)

        return x


@dataclass
class ClassifierArgs:
    """
    This dataclass defines a training configuration.
    feel free to add/change it as you see fit, do NOT remove the following fields as we will use
    them in test time.
    If you add additional values to your training configuration please add them in here with
    default values (so run won't break when we test this).
    """
    # we will use this to give an absolute path to the data, make sure you read the data using this argument.
    # you may assume the train data is the same
    path_to_training_data_dir: str = "./an4/train/an4/"

    # you may add other args here
    path_to_test_data_dir: str = './an4/test/an4/'

    kernels_per_layer = [16, 32, 64, 64, 64, 128, 256]


class AsrModel:

    def __init__(self, args: ClassifierArgs, net: CharacterDetectionNet):
        self.charater_detector = net
        pass

    def general_classification(self, audio_files: tp.Union[tp.List[str], torch.Tensor], method: str) -> tp.List[
        int]:
        """
        function to classify a given audio using method distance
        audio_files: list of audio file paths or a batch of audio files of shape [Batch, Channels, Time]
        method: the method to calculate distance
        return: list of predicted results for each entry in the batch
        """
        if isinstance(audio_files, list) or isinstance(audio_files, str):
            audio_files = utils.load_wav_files(audio_files)
        elif isinstance(audio_files, torch.Tensor):
            # Average over channels
            audio_files = audio_files.mean(dim=1)
        audio_mfcc = utils.extract_mfcc(audio_files)

        predicted_labels = []
        for audio in audio_mfcc:
            predicted_labels.append(self.predict(audio))

        return predicted_labels

    @abstract
    def predict(self, wav):
        pass


Class


# train NW
def train(nn):
    pass


if __name__ == '__main__':
    pass
    # utils.extract_mfccs()

print("Hellow Neriya!")

"""
load mp3
extract mel spec
write NN
build training function
    plot function, to plot the test acc (overfiting problem)

"""
