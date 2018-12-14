import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from scipy.linalg import sqrtm

class Conv1DBlock(nn.Module):
    '''
        Implements a few layers for 1D Convoloution
        Input: (batch_size x 1 x input_size) where input_size >= 256 and input_size % 2 == 0
        Output: (batch_size x (first_conv_channels * 4) * (input_size / 8))

        E.g. 1*2048 -> 16*1024 -> 32*512 -> 64*256
    '''
    def __init__(self, input_size=2048, first_conv_channels=16, final_layer_prelu=False):
        super(Conv1DBlock, self).__init__()
        if input_size < 256 or input_size % 2 != 0:
            raise NotImplementedError("Please choose a valid input size")
        
        self.input_size = input_size        
        self.main_layer = nn.Sequential(
                                nn.Conv1d(1, first_conv_channels, 5, padding=2),
                                nn.PReLU(),
                                nn.MaxPool1d(2, stride=2),

                                nn.Conv1d(first_conv_channels, first_conv_channels*2, 5, padding=2),
                                nn.PReLU(),
                                nn.MaxPool1d(2, stride=2),

                                nn.Conv1d(first_conv_channels*2, first_conv_channels*4, 5, padding=2),
                                nn.PReLU(),
                                nn.MaxPool1d(2, stride=2),
        )
        if final_layer_prelu:
            self.main_layer.add_module("PReLU_Final", nn.PReLU())
    
    def forward(self, x):
        return self.main_layer(x.view(-1, 1, self.input_size))


class FCNBlock(nn.Module):
    '''
        Implements a few layers for a Fully Connected Network / Multi-layer Perceptron
        Input: (batch_size x input_size)
        Output: (batch_size x (first_layer_features / 2))

        E.g 2048 -> 4096 -> 2048 -> 1024
    '''
    def __init__(self, input_size):
        super(FCNBlock, self).__init__()
        self.input_size = input_size

        self.main_layer = nn.Sequential(
                                nn.Linear(input_size, input_size*2),    # 256 x 64
                                nn.ReLU(),
                                nn.Linear(input_size*2, input_size),
                                nn.ReLU(),
                                nn.Linear(input_size, int(input_size/2)),
        )
    
    def forward(self, x):
        return self.main_layer(x)



class SimpleEmbeddingNetV1(nn.Module):
    def __init__(self, input_size):
        super(SimpleEmbeddingNetV1, self).__init__()
        self.input_size = input_size

        self.main_layer =  FCNBlock(input_size=input_size)
    
    def forward(self, x):
        return self.main_layer(x.view(-1, self.input_size))


class SimpleEmbeddingNetV2(nn.Module):
    def __init__(self, input_size, first_conv_channels=16, out_features=512):
        super(SimpleEmbeddingNetV2, self).__init__()
        self.input_size = input_size

        self.conv_layer =   Conv1DBlock(input_size,
                                        first_conv_channels=first_conv_channels,
                                        final_layer_prelu=True)

        self.fc_layer =     nn.Linear(int(first_conv_channels*4 * input_size/8), out_features)

    
    def forward(self, x):
        conv_out = self.conv_layer(x.view(-1, 1, self.input_size))
        return self.fc_layer(conv_out.view(conv_out.shape[0], -1))

