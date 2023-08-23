import torch
import torch.nn as nn
import utils

index2char, char2index = utils.create_index(['@'])


# basic CNN version

class CharNet1(nn.Module):
    def __init__(self, classifierArgs):
        super().__init__()
        self.name = 'CharNet1'
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

# CNN version with batch normalization

class CharNet_1_BN(nn.Module):
    """CharNet + batch norm"""
    def __init__(self, classifierArgs):
        super().__init__()
        self.name = "CharNet1_BN"

        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        conv_kernels = classifierArgs.kernels_per_layer
        self.conv0 = nn.Conv2d(1, conv_kernels[0], 3, padding=1)
        self.bn0 = nn.BatchNorm2d(conv_kernels[0])
        self.conv1 = nn.Conv2d(conv_kernels[1 - 1], conv_kernels[1], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_kernels[1])
        self.conv2 = nn.Conv2d(conv_kernels[2 - 1], conv_kernels[2], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_kernels[2])
        self.conv3 = nn.Conv2d(conv_kernels[3 - 1], conv_kernels[3], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_kernels[3])
        self.conv4 = nn.Conv2d(conv_kernels[4 - 1], conv_kernels[4], 3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_kernels[4])
        self.conv5 = nn.Conv2d(conv_kernels[5 - 1], conv_kernels[5], 3, padding=1)
        self.bn5 = nn.BatchNorm2d(conv_kernels[5])
        self.conv6 = nn.Conv2d(conv_kernels[6 - 1], conv_kernels[6], 3, padding=1)
        self.bn6 = nn.BatchNorm2d(conv_kernels[6])
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 1))
        self.linear = nn.Linear(1024, len(index2char))

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpooling(self.relu(self.bn2(self.conv2(x))))
        x = self.maxpooling(self.relu(self.bn3(self.conv3(x))))
        x = self.maxpooling(self.relu(self.bn4(self.conv4(x))))
        x = self.maxpooling(self.relu(self.bn5(self.conv5(x))))
        x = self.maxpooling(self.relu(self.bn6(self.conv6(x))))
        x = self.flatten(x)
        x = x.permute(2, 0, 1)
        x = self.linear(x)
        x = torch.nn.functional.log_softmax(x, dim=2)
        return x


class ShallowMFCCCharNet1BN(nn.Module):
    def __init__(self, classifierArgs):
        super().__init__()
        self.name = "SHALLOW_MFCC_CharNet"

        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        conv_kernels = classifierArgs.kernels_per_layer
        self.conv0 = nn.Conv2d(1, conv_kernels[0], 3, padding=1)
        self.bn0 = nn.BatchNorm2d(conv_kernels[0])
        self.conv1 = nn.Conv2d(conv_kernels[1 - 1], conv_kernels[1], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_kernels[1])
        self.conv2 = nn.Conv2d(conv_kernels[2 - 1], conv_kernels[2], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_kernels[2])
        self.conv3 = nn.Conv2d(conv_kernels[3 - 1], conv_kernels[3], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_kernels[3])
        self.conv4 = nn.Conv2d(conv_kernels[4 - 1], conv_kernels[4], 3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_kernels[4])
        self.conv5 = nn.Conv2d(conv_kernels[5 - 1], conv_kernels[5], 3, padding=1)
        self.bn5 = nn.BatchNorm2d(conv_kernels[5])
        self.conv6 = nn.Conv2d(conv_kernels[6 - 1], conv_kernels[6], 3, padding=1)
        self.bn6 = nn.BatchNorm2d(conv_kernels[6])
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 1))
        self.linear = nn.Linear(512, len(index2char))

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.maxpooling(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpooling(self.relu(self.bn5(self.conv5(x))))
        x = self.maxpooling(self.relu(self.bn6(self.conv6(x))))
        x = self.flatten(x)
        x = x.permute(2, 0, 1)
        x = self.linear(x)
        x = torch.nn.functional.log_softmax(x, dim=2)
        return x


class MFCCCharNet1BN(nn.Module):
    def __init__(self, classifierArgs):
        super().__init__()
        self.name = "MFCC_CharNet"

        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        conv_kernels = classifierArgs.kernels_per_layer
        self.conv0 = nn.Conv2d(1, conv_kernels[0], 3, padding=1)
        self.bn0 = nn.BatchNorm2d(conv_kernels[0])
        self.conv1 = nn.Conv2d(conv_kernels[1 - 1], conv_kernels[1], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_kernels[1])
        self.conv2 = nn.Conv2d(conv_kernels[2 - 1], conv_kernels[2], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_kernels[2])
        self.conv3 = nn.Conv2d(conv_kernels[3 - 1], conv_kernels[3], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_kernels[3])
        self.conv4 = nn.Conv2d(conv_kernels[4 - 1], conv_kernels[4], 3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_kernels[4])
        self.conv5 = nn.Conv2d(conv_kernels[5 - 1], conv_kernels[5], 3, padding=1)
        self.bn5 = nn.BatchNorm2d(conv_kernels[5])
        self.conv6 = nn.Conv2d(conv_kernels[6 - 1], conv_kernels[6], 3, padding=1)
        self.bn6 = nn.BatchNorm2d(conv_kernels[6])
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 1))
        self.linear = nn.Linear(512, len(index2char))

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpooling(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpooling(self.relu(self.bn5(self.conv5(x))))
        x = self.maxpooling(self.relu(self.bn6(self.conv6(x))))
        x = self.flatten(x)
        x = x.permute(2, 0, 1)
        x = self.linear(x)
        x = torch.nn.functional.log_softmax(x, dim=2)
        return x


class DeeperMFCCCharNet1BN(nn.Module):
    def __init__(self, classifierArgs):
        super().__init__()
        self.name = "DEEPER_MFCC_CharNet"

        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        conv_kernels = classifierArgs.kernels_per_layer
        self.conv0 = nn.Conv2d(1, conv_kernels[0], 3, padding=1)
        self.bn0 = nn.BatchNorm2d(conv_kernels[0])
        self.conv1 = nn.Conv2d(conv_kernels[1 - 1], conv_kernels[1], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_kernels[1])
        self.conv2 = nn.Conv2d(conv_kernels[2 - 1], conv_kernels[2], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_kernels[2])
        self.conv3 = nn.Conv2d(conv_kernels[3 - 1], conv_kernels[3], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_kernels[3])
        self.conv4 = nn.Conv2d(conv_kernels[4 - 1], conv_kernels[4], 3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_kernels[4])
        self.conv5 = nn.Conv2d(conv_kernels[5 - 1], conv_kernels[5], 3, padding=1)
        self.bn5 = nn.BatchNorm2d(conv_kernels[5])
        self.conv6 = nn.Conv2d(conv_kernels[6 - 1], conv_kernels[6], 3, padding=1)
        self.bn6 = nn.BatchNorm2d(conv_kernels[6])
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 1))
        self.linear = nn.Linear(512, len(index2char))

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpooling(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.maxpooling(self.relu(self.bn5(self.conv5(x))))
        x = self.maxpooling(self.relu(self.bn6(self.conv6(x))))
        x = self.flatten(x)
        x = x.permute(2, 0, 1)
        x = self.linear(x)
        x = torch.nn.functional.log_softmax(x, dim=2)
        return x


class Deeper2MFCCCharNet1BN(nn.Module):
    def __init__(self, classifierArgs):
        super().__init__()
        self.name = "DEEPER2_MFCC_CharNet"

        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        conv_kernels = classifierArgs.kernels_per_layer
        self.conv0 = nn.Conv2d(1, conv_kernels[0], 3, padding=1)
        self.bn0 = nn.BatchNorm2d(conv_kernels[0])
        self.conv1 = nn.Conv2d(conv_kernels[1 - 1], conv_kernels[1], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_kernels[1])
        self.conv2 = nn.Conv2d(conv_kernels[2 - 1], conv_kernels[2], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_kernels[2])
        self.conv3 = nn.Conv2d(conv_kernels[3 - 1], conv_kernels[3], 3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_kernels[3])
        self.conv4 = nn.Conv2d(conv_kernels[4 - 1], conv_kernels[4], 3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_kernels[4])
        self.conv4p5 = nn.Conv2d(conv_kernels[4], 32, 3, padding=1)
        self.bn4p5 = nn.BatchNorm2d(32)
        self.convR4p5 = nn.Conv2d(32,conv_kernels[4], 3, padding=1)
        self.bn4Rp5 = nn.BatchNorm2d(conv_kernels[4])
        self.conv5 = nn.Conv2d(conv_kernels[5 - 1], conv_kernels[5], 3, padding=1)
        self.bn5 = nn.BatchNorm2d(conv_kernels[5])
        self.conv6 = nn.Conv2d(conv_kernels[6 - 1], conv_kernels[6], 3, padding=1)
        self.bn6 = nn.BatchNorm2d(conv_kernels[6])
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 1))
        self.linear = nn.Linear(512, len(index2char))

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpooling(self.relu(self.bn3(self.conv3(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn4p5(self.conv4p5(x)))
        x = self.relu(self.bn4Rp5(self.convR4p5(x)))
        x = self.maxpooling(self.relu(self.bn5(self.conv5(x))))
        x = self.maxpooling(self.relu(self.bn6(self.conv6(x))))
        x = self.flatten(x)
        x = x.permute(2, 0, 1)
        x = self.linear(x)
        x = torch.nn.functional.log_softmax(x, dim=2)
        return x


# class Ansemble():
#     def __init__(self, nets, weights):
#         self.nets = nets
#         self.weights = weights
#
#     def predict(self, input):
#         outputs = []
#         outputs = torch.Tensor(())
