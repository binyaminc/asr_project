import string
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchvision
from torchvision import transforms

from torch import optim
from torch.utils.data import DataLoader, Dataset
import librosa
import matplotlib.pyplot as plt
# import tp
from jiwer import wer
import utils
import os

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


class CustomASRDataset(Dataset):
    def __init__(self, audio_dir, label_dir):
        self.audio_dir = audio_dir
        audio_data = utils.load_wav_files(audio_dir)

        # in case of working with the spectrogram, split the spectrogram
        self.specs = torch.split(audio_data, split_size_or_sections=1470, dim=1)
        # in case of working with mfcc features
        # audio_data = utils.load_wav_files(audio_dir)

        self.label_dir = label_dir
        self.file_list = os.listdir(audio_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load the corresponding label (assuming the label file has the same name as the audio file)
        audio_filename = self.file_list[idx]
        label_filename = os.path.splitext(audio_filename)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)

        with open(label_path, 'r') as label_file:
            label = label_file.read().strip()

        # find the spectrogram 
        spectrogram = self.specs[idx]

        return spectrogram, label


def main():
    # define the network
    net = CharacterDetectionNet(ClassifierArgs())

    # Define the CTC loss
    ctc_loss = nn.CTCLoss()

    training_dataset = CustomASRDataset(ClassifierArgs.training_path + '\\wav', train_path + '\\txt')
    training_loader = DataLoader(training_dataset, batch_size=ClassifierArgs.batch_size, shuffle=True)

    validation_dataset = CustomASRDataset(ClassifierArgs.val_path + '\\wav', ClassifierArgs.val_path + '\\txt')
    validation_loader = DataLoader(validation_dataset, batch_size=ClassifierArgs.batch_size, shuffle=True)

    # Set up the training loop
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    epochs = 30

    train_loss, validation_loss = [], []

    early_stopper = EarlyStopper(patience=3, min_delta=10)
    for epoch in np.arange(epochs):
        train_loss.append(train_one_epoch(ctc_loss, net, optimizer, training_loader))
        validation_loss.append(dataloader_score(net, validation_loader))
        if early_stopper.early_stop(validation_loss):
            break

    # TODO: plt losses
    # TODO:
    # TODO: use test to check the network performance with wer


def train_one_epoch(loss_function, net, optimizer, training_data_loader):
    # Iterate through the training data
    # data=batch,, label
    for spectrogram, target_text, spectrogram_lengths, target_lengths in training_data_loader:  # (batch, spce, splits,labels)
        optimizer.zero_grad()

        # Forward pass
        output = net(spectrogram)

        # TODO: use our ctc loss?
        # loss = loss_function(output, target_text, output_lengths, target_lengths)
        loss = loss_function(output, target_text, target_lengths, target_lengths)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        return loss.item()


def dataloader_score(net, data_loader):
    i = 0
    loss = 0
    with torch.no_grad():
        for item, label in data_loader:
            loss = nn.CTCLoss(net(item), label)
            i += 1
    return loss / i


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
    training_path: str = "./an4/train/an4/"
    val_path: str = "./an4/val/an4/"
    path_to_test_data_dir: str = "./an4/test/an4/"

    kernels_per_layer = [16, 32, 64, 64, 64, 128, 256]
    batch_size = 32


class AsrModel:

    def __init__(self, args: ClassifierArgs, net: CharacterDetectionNet):
        self.charater_detector = net
        pass

    def general_classification(self, audio_files, method: str):
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

    # @abstract
    def predict(self, wav):
        pass


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


if __name__ == '__main__':
    main()
    # utils.extract_mfccs()

print("Hellow Neriya!")

"""
load mp3
extract mel spec
write NN
build training function
    plot function, to plot the test acc (overfiting problem)

"""
