from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from jiwer import wer, cer
import os
import matplotlib.pyplot as plt
from networks import index2char, char2index
from networks import *
import utils
from tqdm import tqdm
import pandas as pd

# from torchaudio.models.decoder import cuda_ctc_decoder

train_path = r'an4\\train\\an4\\'
DATASET_STATES = ['WAVEFORM', 'MFC', 'MFCC']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    test_path: str = "./an4/test/an4/"

    kernels_per_layer = [16, 32, 64, 64, 64, 128, 256]
    batch_size = 32
    epochs = 50
    save_model = True
    stage = 'Stage2/'


def hash_label(label: str):
    return torch.tensor([char2index[c] for c in label.lower()])


class CustomASRDataset(Dataset):
    def __init__(self, audio_dir, label_dir, frame_length, input_type='MFC', is_training=False):
        # as input type defines the pre processing of the data.
        assert input_type in DATASET_STATES
        self.audio_dir = audio_dir
        self.audio_data, self.input_length = utils.load_wav_files(audio_dir, input_type, is_training)
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
        self.is_training = is_training

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_filename = self.file_list[idx]
        label_filename = os.path.splitext(audio_filename)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)

        with open(label_path, 'r') as label_file:
            label = label_file.read().strip()

        # Spectrogram is splitted to 128 values per time step. time step decided by max length at _init,load_wav...
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


def main():
    t = ClassifierArgs()
    data_state = DATASET_STATES[1]

    training_dataset = CustomASRDataset(t.training_path + '\\wav', train_path + '\\txt', 128, is_training=True)
    training_loader = DataLoader(training_dataset, batch_size=t.batch_size, shuffle=True)

    validation_dataset = CustomASRDataset(t.val_path + '\\wav', t.val_path + '\\txt', 128)
    validation_loader = DataLoader(validation_dataset, batch_size=t.batch_size, shuffle=False)

    test_dataset = CustomASRDataset(t.test_path + '\\wav', t.test_path + '\\txt', 128)
    test_loader = DataLoader(test_dataset, batch_size=t.batch_size, shuffle=False)

    print(f"device: {device}")

    ctc_loss = nn.CTCLoss()

    # for net in [ExtendedCharNet1(t), CharNet_1(t), CharNet_1_BN(t), CharNet_plusConv_BN(t)]:
    for net in [ExtendedCharNet1(t)]:

        optimizer = optim.Adam(net.parameters(), lr=0.0005)
        early_stopper = EarlyStopper(patience=5, min_delta=0.01)

        print("data loaded. start training")
        train_ctc_losses, val_ctc_losses, test_ctc_losses = [], [], []
        val_ctc_losses, val_wer_losses, val_cer_losses = [], [], []

        net.to(device)
        for epoch in tqdm(np.arange(ClassifierArgs.epochs)):
            train_ctc_loss = train_one_epoch(ctc_loss, net, optimizer, training_loader)
            train_ctc_losses.append(train_ctc_loss)

            val_ctc_loss, val_wer_loss, val_cer_loss = dataloader_score(ctc_loss, net, validation_loader)
            val_ctc_losses.append(val_ctc_loss)
            val_wer_losses.append(val_wer_loss)
            val_cer_losses.append(val_cer_loss)
            print(
                f'\nepoch {epoch}: TRAIN loss = {round(train_ctc_loss, 6)} , VAL loss {round(val_ctc_loss, 6)}')

            if (early_stopper.early_stop(
                    val_cer_loss) or epoch == ClassifierArgs.epochs - 1) and ClassifierArgs.save_model:
                torch.save(net.state_dict(), f'saved_models/{net.name}{data_state}_epoch{epoch}.pt')
                if epoch != ClassifierArgs.epochs - 1:
                    print("exit early")
                break

        val_cer_losses = np.clip(np.array(val_cer_losses), a_min=0, a_max=1)
        val_wer_losses = np.clip(np.array(val_wer_losses), a_min=0, a_max=1)

        title = f'{ClassifierArgs.stage}{net.name}'
        # can be shortened to a loop, later on.
        plot_name = 'ctc loss'  # f'{net.name}_{data_state}_ctc'
        plotter(title+' '+plot_name, plot_name=title+' '+plot_name, x_axis_label='epochs', y_axis_label='loss',
                data=[train_ctc_losses, val_ctc_losses],
                data_labels=['training loss', 'val loss'])

        plotter('Validation Wer and Cer', title, x_axis_label='epochs', y_axis_label='score',
                data=[val_wer_losses, val_cer_losses], data_labels=['wer score', 'cer score'])

        # tuples of ctc wer ser
        test_scores_ctc_wer_ser = torch.Tensor(dataloader_score(ctc_loss, net, test_loader))
        train_scores_ctc_wer_ser = torch.Tensor(dataloader_score(ctc_loss, net, training_loader))
        val_scores_ctc_wer_ser = torch.Tensor(dataloader_score(ctc_loss, net, validation_loader))
        all_scores = torch.stack((train_scores_ctc_wer_ser, val_scores_ctc_wer_ser, test_scores_ctc_wer_ser)).numpy()
        title = title.replace('_', ' ')
        pd.DataFrame(all_scores).to_csv(f'{title} results.csv')


def plotter(title, plot_name, x_axis_label, y_axis_label, data, data_labels):
    # plt losses
    for i in range(len(data)):
        plt.plot(data[i], label=data_labels[i])
    plt.legend()
    plt.ylabel(y_axis_label)
    plt.xlabel(x_axis_label)
    # plt.title(f'{type(net)} data preprocessing {data_state} full')
    if plot_name != title:
        plt.suptitle(title, fontsize=18)
    plt.title(f'{plot_name}')
    plt.savefig(f'plots/' + plot_name + '.jpeg')
    plt.clf()
    plt.cla()


def train_one_epoch(loss_function, net, optimizer, training_data_loader):
    sum_ctc_loss, sum_wer_loss, sum_cer_loss = 0, 0, 0
    i = 0

    torch.enable_grad()
    # Iterate through the training data
    for specs, target_text, spectrogram_lengths, target_lengths in training_data_loader:  # (batch, spce, splits,labels)
        optimizer.zero_grad()

        # instead of the shape (32, 128, 276), I want a shape (32, 1, 128, 276) s.t. the input will have 1 channels.
        specs = torch.unsqueeze(specs, 1).to(device)

        # Forward pass
        output = net(specs)
        target_text.to(device)
        spectrogram_lengths.to(device)
        target_lengths.to(device)

        # compute loss
        ctc_loss = loss_function(output, target_text, spectrogram_lengths, target_lengths)
        sum_ctc_loss += ctc_loss.item()

        # Backward pass and optimization
        ctc_loss.backward()
        optimizer.step()

        i += 1
        break

    torch.cuda.empty_cache()
    return sum_ctc_loss / i


def dataloader_score(loss_function, net, data_loader):
    sum_ctc_loss, sum_wer_loss, sum_cer_loss = 0, 0, 0
    i = 0

    with torch.no_grad():
        for specs, target_text, spectrogram_lengths, target_lengths in data_loader:
            # add dimension for the channels
            specs = torch.unsqueeze(specs, 1).to(device)

            # Forward pass
            output = net(specs)
            target_text.to(device)
            spectrogram_lengths.to(device)
            target_lengths.to(device)

            # compute ctc loss
            ctc_loss = loss_function(output, target_text, spectrogram_lengths, target_lengths)
            sum_ctc_loss += ctc_loss.item()
            # compute er loss
            wer_loss, cer_loss = get_er_loss(output, target_text)
            sum_wer_loss += wer_loss
            sum_cer_loss += cer_loss
            i += 1
            if i > 100: break  # after increasing training set size, it is endless

    return sum_ctc_loss / i, sum_wer_loss / i, sum_cer_loss / i


def get_er_loss(output, target_text):
    # convert output to (batch_size, time_slices, characters)
    output = output.permute(1, 0, 2)

    wer_losses_sum, cer_losses_sum = 0, 0

    target_text = target_text.detach().numpy()
    k = 0
    for (i, probs) in enumerate(output):
        n_sentences = beam_search(probs, n=3)

        best_sentence = n_sentences[-1]
        best_sentence = ''.join([index2char[c] for c in best_sentence])
        best_sentence = best_sentence.replace('@', '').replace('<BLANK>', '')

        curr_reference = ''.join([index2char[c] for c in target_text[i]])
        curr_reference = curr_reference.replace('@', '')

        # calc wer and cer loss
        wer_loss = wer(reference=curr_reference, hypothesis=best_sentence)
        cer_loss = cer(reference=curr_reference, hypothesis=best_sentence)

        # wer_loss, cer_loss = 0, 0
        wer_losses_sum += wer_loss
        cer_losses_sum += cer_loss

        k += 1

    return wer_losses_sum / (k), cer_losses_sum / (k)


def beam_search(probs, n=3):
    """
    n - beam width, the amount of trails I follow in each step.
    """
    # convert log_softmax to regular softmax, so that we can add and multiple normally
    probs = torch.exp(probs)
    probs = probs.detach().cpu().numpy()

    texts = {}

    # initialize dictionary with the first time slice
    step0_probs = probs[0]
    biggest_idxes = sorted(range(len(step0_probs)), key=lambda p: step0_probs[p])[-n:]  # TODO: could be more efficient
    for idx in biggest_idxes:
        texts[(idx,)] = [step0_probs[idx], 0]  # example: texts[[a]] = [P([a]), P([a, <e>])=0]
    if (0,) in texts:
        texts[(0,)] = [0, texts[(0,)][0]]  # TODO: not sure is necessary

    for step_probs in probs[1:]:
        new_texts = {}

        for text, trail_probs in texts.items():
            new_texts = add_step_to_trail(new_texts, text, trail_probs, step_probs)

        # find the n entries with the highest probability
        texts = dict(sorted(new_texts.items(), key=lambda item: item[1][0] + item[1][1])[
                     -n:])  # TODO: is the best in the 0 or last position?

    return list(texts.keys())


def add_step_to_trail(texts, trail, prob, step_probs):
    for (char, char_prob) in enumerate(step_probs):

        # new char is the same as the last char in trail
        if char == trail[-1]:
            if trail not in texts: texts[trail] = [0, 0]
            if not trail + (char,) in texts: texts[trail + (char,)] = [0, 0]

            texts[trail][0] += prob[0] * char_prob
            texts[trail + (char,)][1] += prob[1] * char_prob

        # new char is epsilon
        elif char == 0:
            if not trail in texts: texts[trail] = [0, 0]

            texts[trail][1] += prob[0] * char_prob + prob[1] * char_prob

        # any other chars
        else:
            if not trail + (char,) in texts: texts[trail + (char,)] = [0, 0]

            texts[trail + (char,)][0] += prob[0] * char_prob + prob[1] * char_prob

    return texts


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
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


def checkplot():
    net = CharNet_1(ClassifierArgs())
    net.load_state_dict(torch.load(r'C:\work\projects\asr_project\saved_models\CharNet1_MFC_epoch_49.pt'))

    training_dataset = CustomASRDataset(ClassifierArgs.training_path + '\\wav', train_path + '\\txt', 128)
    training_loader = DataLoader(training_dataset, batch_size=ClassifierArgs.batch_size, shuffle=True)

    for specs, target_text, spectrogram_lengths, target_lengths in training_loader:  # Use correct variable name
        specs = specs.unsqueeze(dim=1)
        with torch.no_grad():  # Correct the usage of torch.no_grad()
            out = net(specs)  # Pass the correct input (specs) to the model
            utils.plot_CTC_output(out[:, 0, :])  # Assuming you have a function to plot CTC outputs
        break


if __name__ == '__main__':
    main()

    # checkplot()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    """to plot barplot"""
    # CSV data as a string

    # df = pd.read_csv(r'C:\work\projects\asr_project_2\Stage 2 CharNet1 ctc loss results.csv')
    # # Set the 'Unnamed: 0' column as the index
    # df.set_index('Unnamed: 0', inplace=True)
    #
    # # Create a bar plot
    # ax = df.plot(kind='bar', figsize=(10, 6))
    #
    # # Set labels and title
    # plt.xlabel('Data Split')
    # plt.ylabel('Value')
    # plt.title('Comparison of ctc, wer, and cer')
    #
    # # Show the plot
    # plt.show()
"""
- basic model
1 batch norm                            V
2 WER metric                            ?
3 to change both models to mfcc         ?
4 to change both models to waveform     ?
5 transformers
"""
