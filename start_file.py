import string
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
# from jiwer import wer
import utils
import os
# from nets import *
import matplotlib.pyplot as plt
# from tqdm import tqdm
import glob
import time

index2char, char2index = utils.create_index(['@'])
train_path = r'an4\\train\\an4\\'


class CharacterDetectionNet(nn.Module):
    def __init__(self, classifierArgs):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        conv_kernels = classifierArgs.kernels_per_layer
        self.conv0 = nn.Conv2d(1, conv_kernels[0], 3, padding=1)
        self.conv1 = nn.Conv2d(conv_kernels[1 - 1], conv_kernels[1], 3, padding=1)
        self.conv2 = nn.Conv2d(conv_kernels[2 - 1], conv_kernels[2], 3, padding=1)
        self.conv3 = nn.Conv2d(conv_kernels[3 - 1], conv_kernels[3], 3, padding=1)
        self.conv4 = nn.Conv2d(conv_kernels[4 - 1], conv_kernels[4], 3, padding=1)
        self.conv5 = nn.Conv2d(conv_kernels[5 - 1], conv_kernels[5], 3, padding=1)
        self.conv6 = nn.Conv2d(conv_kernels[6 - 1], conv_kernels[6], 3, padding=1)
        self.relu = nn.ReLU()
        # self.softmax = torch.nn.functional.log_softmax(dim=2)
        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 1))
        self.linear = nn.Linear(1024, len(index2char))

        # a more pythonic way to go about this:
        # self.convs = [nn.Conv1d(1, conv_kernels[0], 3, padding=1)] + \
        #              [nn.Conv1d(conv_kernels[i - 1], conv_kernels[i], 3, padding=1) for i in
        #               range(1, len(conv_kernels))]

    def forward(self, x):
        # print(x.shape)
        x = x.permute(0, 1, 3, 2)
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.maxpooling(self.relu(self.conv2(x)))
        x = self.maxpooling(self.relu(self.conv3(x)))
        x = self.maxpooling(self.relu(self.conv4(x)))
        x = self.maxpooling(self.relu(self.conv5(x)))
        x = self.maxpooling(self.relu(self.conv6(x)))
        x = self.flatten(x)
        x = x.permute(2, 0, 1)
        x = self.linear(x)
        # x = self.softmax(x)
        x = torch.nn.functional.log_softmax(x, dim=2)
        return x


def hash_label(label: str):
    return torch.tensor([char2index[c] for c in label.lower()])


class CustomASRDataset(Dataset):
    def __init__(self, audio_dir, label_dir, frame_length):
        self.audio_dir = audio_dir
        self.audio_data, self.input_length = utils.load_wav_files(audio_dir)
        self.label_dir = label_dir
        self.file_list = os.listdir(audio_dir)
        self.file_list = [x for x in os.listdir(audio_dir) if x.endswith('.wav')]
        self.frame_length = frame_length

        # save the max len of label
        txt_file_list = os.listdir(label_dir)
        maxl = 0
        for label in txt_file_list:
            with open(label_dir + '\\\\' + label, 'r') as label_file:
                label = label_file.read().strip()
                if len(label) > maxl: maxl = len(label)
        self.max_len_label = maxl

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_filename = self.file_list[idx]
        label_filename = os.path.splitext(audio_filename)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)

        with open(label_path, 'r') as label_file:
            label = label_file.read().strip()

        # Spectrogram is splitted to 128 values per time step. time step decided by max length at __init__,load_wav_...
        spectrogram = self.audio_data[idx]
        pad = self.max_len_label - len(label)
        label = label + pad * '@'
        # goal is to return this: spectrogram, target_text, spectrogram_lengths, target_lengths
        return spectrogram, hash_label(label), self.input_length[idx], len(label) - pad


def custom_collate_fn(batch):
    spectrogram_frames, labels = zip(*batch)

    # Pad the spectrogram frames to the length of the longest frame in the batch
    max_frame_length = max(frame.shape[1] for frame in spectrogram_frames)
    padded_spectrogram_frames = torch.stack(
        [torch.nn.functional.pad(frame, (0, max_frame_length - frame.shape[1])) for frame in spectrogram_frames]
    )

    # Return the padded spectrogram frames and original labels
    return padded_spectrogram_frames, labels


epochs = 100


def main():
    # define the network
    net = CharacterDetectionNet(ClassifierArgs())
    net.to(device)

    # Define the CTC loss
    ctc_loss = nn.CTCLoss()

    training_dataset = CustomASRDataset(ClassifierArgs.training_path + '\\wav', train_path + '\\txt', 128)
    training_loader = DataLoader(training_dataset, batch_size=ClassifierArgs.batch_size, shuffle=False)

    validation_dataset = CustomASRDataset(ClassifierArgs.val_path + '\\wav', ClassifierArgs.val_path + '\\txt', 128)
    validation_loader = DataLoader(validation_dataset, batch_size=ClassifierArgs.batch_size, shuffle=True)

    # Set up the training loop
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    train_loss, validation_loss = [], []

    early_stopper = EarlyStopper(patience=3, min_delta=10)
    for epoch in np.arange(epochs):
        # start_time = time.time()
        train_loss.append(train_one_epoch(ctc_loss, net, optimizer, training_loader))
        print(f"epoch:{epoch} loss:{train_loss[-1]}")
        validation_loss.append(dataloader_score(ctc_loss, net, validation_loader))
        # end_time = time.time()
        # print(end_time - start_time)
        # torch.save(net.state_dict(), f'epoch {epoch}.pt')  # saved_models/_input_{input_size}/d_model_{d_model}/n_heads_{nhead}/n_encoder_{num_encoder_layers}/epoch_{epoch}
        # if early_stopper.early_stop(validation_loss):
        #    break
    print(zip(train_loss, validation_loss))

    # plt losses
    plt.plot(train_loss, label='train')
    plt.plot(validation_loss, label='validation')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('Train and Validation Loss')
    plt.show()


def dataloader_score(loss_function, net, data_loader):
    sum_loss_float = 0
    i = 0
    with torch.no_grad():
        for specs, target_text, spectrogram_lengths, target_lengths in data_loader:
            # add dimension for the channels
            specs = torch.unsqueeze(specs, 1).to(device)

            # Forward pass
            output = net(specs)

            # compute loss
            loss = loss_function(output, target_text, spectrogram_lengths, target_lengths)
            sum_loss_float += loss.item()
            i += 1

    return sum_loss_float / i


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # todo add cuda


def train_one_epoch(loss_function, net, optimizer, training_data_loader):
    sum_loss_float = 0
    i = 0

    torch.enable_grad()
    # Iterate through the training data
    for specs, target_text, spectrogram_lengths, target_lengths in training_data_loader:  # (batch, spce, splits,labels)
        optimizer.zero_grad()

        # instead of the shape (32, 128, 276), I want a shape (32, 1, 128, 276) s.t. the input will have 1 channels.
        specs = torch.unsqueeze(specs, 1).to(device)

        # Forward pass
        output = net(specs).to(device)
        target_text.to(device)
        spectrogram_lengths.to(device)
        target_lengths.to(device)
        # loss = loss_function(output, target_text, output_lengths, target_lengths)
        loss = loss_function(output, target_text, spectrogram_lengths, target_lengths)
        # print(f"loss after batch {i}: {loss.item()}")
        sum_loss_float += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        i += 1
        break

    torch.cuda.empty_cache()
    return sum_loss_float / i


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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

"""
load mp3
extract mel spec
write NN
build training function
    plot function, to plot the test acc (overfiting problem)


A) check for gpu
B) try to overfit a small part of the net, to make sure the function the net
"""
