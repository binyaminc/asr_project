from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from jiwer import wer
import os
import matplotlib.pyplot as plt

from networks import index2char, char2index
from networks import CharacterDetectionNet_1, CharacterDetectionNet_1_batch_normed
import utils

train_path = r'an4\\train\\an4\\'
epochs = 100
DATASET_STATES = {'WAVEFORM', 'MFC', 'MFCC'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hash_label(label: str):
    return torch.tensor([char2index[c] for c in label.lower()])


class CustomASRDataset(Dataset):
    def __init__(self, audio_dir, label_dir, frame_length, input_type='MFC'):
        # as input type defines the pre processing of the data.
        assert input_type in DATASET_STATES
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


def main():
    # define the network
    # net = CharacterDetectionNet_1(ClassifierArgs())
    net = CharacterDetectionNet_1_batch_normed(ClassifierArgs())

    # Define the CTC loss
    ctc_loss = nn.CTCLoss()

    training_dataset = CustomASRDataset(ClassifierArgs.training_path + '\\wav', train_path + '\\txt', 128)
    training_loader = DataLoader(training_dataset, batch_size=ClassifierArgs.batch_size, shuffle=False)

    validation_dataset = CustomASRDataset(ClassifierArgs.val_path + '\\wav', ClassifierArgs.val_path + '\\txt', 128)
    validation_loader = DataLoader(validation_dataset, batch_size=ClassifierArgs.batch_size, shuffle=True)

    # Set up the training loop
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    train_CTC_losses, val_CTC_losses = [], []
    train_WER_losses, val_WER_losses = [], []

    early_stopper = EarlyStopper(patience=3, min_delta=10)

    net.to(device)
    for epoch in np.arange(epochs):
        # start_time = time.time()
        train_CTC_loss, train_WER_loss = train_one_epoch(ctc_loss, net, optimizer, training_loader)
        train_CTC_losses.append(train_CTC_loss)
        train_WER_losses.append(train_WER_loss)
        print(f"epoch {epoch}: ctc loss = {train_CTC_loss}, wer loss = {train_WER_loss}")

        val_CTC_loss, val_WER_loss = dataloader_score(ctc_loss, net, validation_loader)
        val_CTC_losses.append(val_CTC_loss)
        val_WER_losses.append(val_WER_loss)
        # end_time = time.time()
        # print(end_time - start_time)
        # if epoch == epochs - 1:
        #     torch.save(net.state_dict(),
        #                f'{type(net)}_epoch{epoch}_t_loss_{train_losses[-1]}_v_loss{validation_losses[-1]}.pt')  # saved_models/_input_{input_size}/d_model_{d_model}/n_heads_{nhead}/n_encoder_{num_encoder_layers}/epoch_{epoch}
        # if early_stopper.early_stop(validation_losses):
    print(zip(train_CTC_losses, val_CTC_losses))

    # plt losses
    plt.plot(train_CTC_losses, label='train')
    plt.plot(val_CTC_losses, label='validation')
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('Train and Validation Loss')
    plt.show()


def train_one_epoch(loss_function, net, optimizer, training_data_loader):
    sum_ctc_loss, sum_wer_loss = 0, 0
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
        wer_loss = get_wer_loss(output, target_text)  # target_lengths?
        sum_wer_loss += wer_loss

        # Backward pass and optimization
        ctc_loss.backward()
        optimizer.step()

        i += 1
        # break

    torch.cuda.empty_cache()
    return sum_ctc_loss / i, sum_wer_loss / i


def dataloader_score(loss_function, net, data_loader):
    sum_ctc_loss, sum_wer_loss = 0, 0
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
            # compute wer loss
            wer_loss = get_wer_loss(output, target_text)  # target_lengths?
            sum_wer_loss += wer_loss
            i += 1

    return sum_ctc_loss / i, sum_wer_loss / i


def get_wer_loss(output, target_text):

    # convert output to (batch_size, time_slices, characters)
    output = output.permute(1, 0, 2)

    wer_losses_sum = 0

    target_text = target_text.detach().numpy()

    for (i, probs) in enumerate(output):
        n_sentences = beam_search(probs, n=3)
        best_sentence = n_sentences[-1]
        best_sentence = ''.join([index2char[c] for c in best_sentence])
        best_sentence = best_sentence.replace('@', '').replace('<BLANK>', '')

        curr_reference = ''.join([index2char[c] for c in target_text[i]])
        curr_reference = curr_reference.replace('@', '')

        # calc wer loss
        wer_loss = wer(reference=curr_reference, hypothesis=best_sentence)
        wer_losses_sum += wer_loss
    
    return wer_losses_sum / (i+1)


def beam_search(probs, n=3):
    """
    n - beam width, the amount of trails I follow in each step.
    """
    # convert log_softmax to regular softmax, so that we can add and multiple normally
    probs = torch.exp(probs)
    probs = probs.detach().numpy()
    
    texts = {}

    # initialize dictionary with the first time slice
    step0_probs = probs[0]
    biggest_idxes = sorted(range(len(step0_probs)), key = lambda p: step0_probs[p])[-n:]  # TODO: could be more efficient
    for idx in biggest_idxes:
        texts[(idx,)] = [step0_probs[idx], 0]  # example: texts[[a]] = [P([a]), P([a, <e>])=0]
    if (0,) in texts:
        texts[(0,)] = [0, texts[(0,)][0]] # TODO: not sure is necessary

    for step_probs in probs[1:]:
        new_texts = {}
        
        for text, trail_probs in texts.items():
            new_texts = add_step_to_trail(new_texts, text, trail_probs, step_probs)
            
        # find the n entries with the highest probability
        texts = dict(sorted(new_texts.items(), key=lambda item: item[1][0] + item[1][1])[-n:])  # TODO: is the best in the 0 or last position?

    return list(texts.keys())


def add_step_to_trail(texts, trail, prob, step_probs):

    for (char, char_prob) in enumerate(step_probs):
        
        # new char is the same as the last char in trail
        if char == trail[-1]:
            if not trail in texts: texts[trail] = [0, 0]
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


"""
def canonicalize(trail: str):
    '''
    converts output trail of the NN to readable text.
    example: trail = 'aa<e>b<e>b<e>' (when <e> is epsilon)
             output = 'abb<e>'
    '''
    # going in reversed order to delete doubles
    for i in range(len(trail) - 1, 0, -1):  
        if trail[i] == trail[i - 1]:
            trail = trail[:i] + trail[i + 1:]  # trail.pop(i)

    # going in reversed order to delete epsilon, except for the last epsilon (if exists)
    for i in range(len(trail) - 2, -1, -1):
        if trail[i] == '0': 
            trail = trail[:i] + trail[i + 1:]  # trail.pop(i)
    
    return trail
""" 


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
    batch_size = 8


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
- basic model
1 batch norm                            V
2 WER metric                            ?
3 to change both models to mfcc         ?
4 to change both models to waveform     ?
5 transformers
"""
