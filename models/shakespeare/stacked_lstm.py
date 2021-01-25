import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append('/Users/geoffreywest/Desktop/Research/Srebro/Code/distributed-opt/')
from utils.model_utils import batch_data

class Model(nn.Module):
    '''
    A character prediction LSTM module
    '''
    def __init__(self, seq_len, n_classes, n_hidden, optimizer, seed):
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.n_hidden = n_hidden

        self.encoder = nn.Embedding(input_size, n_hidden) # TODO
        self.lstm = nn.LSTM(n_hidden, n_hidden, n_layers) # TODO
        self.decoder = nn.Linear(n_hidden, n_classes)


    def forward(input, hidden):
