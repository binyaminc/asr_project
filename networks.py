
import torch
import torch.nn as nn
import utils


index2char, char2index = utils.create_index(['@'])

# basic CNN version

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
        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 1))
        self.linear = nn.Linear(1024, len(index2char))

        # a more pythonic way to go about this:
        # self.convs = [nn.Conv1d(1, conv_kernels[0], 3, padding=1)] + \
        #              [nn.Conv1d(conv_kernels[i - 1], conv_kernels[i], 3, padding=1) for i in
        #               range(1, len(conv_kernels))]

    def forward(self, x):
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
        x = torch.nn.functional.log_softmax(x, dim=2)
        return x


