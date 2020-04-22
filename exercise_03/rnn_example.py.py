#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def apply_permutation(tensor, permutation, dim=1):
    # type: (Tensor, Tensor, int) -> Tensor
    return tensor.index_select(dim, permutation)
                 
# 2.1 RNNs as black-box modules (0.5 points)                 
class RNN(nn.Module):
    
    def __init__(self, input_size=20, hidden_size=2, num_layers=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(4, hidden_size)
        self.i2o = nn.Linear(4, 2)  
        
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class GRU(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=8, output_dim=4, n_layers=2, drop_prob=0.2):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.i2h = nn.Linear(8, 2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.i2h(self.relu(out[:,-1]))
        out = self.fc(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim)
        return hidden

class LSTM(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=8, output_dim=4, n_layers=2, drop_prob=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.i2h = nn.Linear(8, 2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:,-1]))
        out = self.i2h(self.relu(out[:,-1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim)
        return hidden




# 2.2 Iterating over an RNN (0.5 points)
class RNNCellBase(nn.Module):
    
    def __init__(self, input_size=20, hidden_size=2, num_layers=2):
        super(RNNCellBase, self).__init__()
        self.rnncell = nn.rnnCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(4, hidden_size)
        self.i2o = nn.Linear(4, 2)  
        
    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        out= self.check_forward_input(input)
        out = self.i2h(out)
        out = self.i2o(out)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            cx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        output = []
        for i in range(input.size(0)):
            hx, cx = nn.rnn(input[i], (hx, cx))
            output.append(hx)
        return output

class GRUCell(nn.Module):
    
    def __init__(self, input_size=20, hidden_size=2, num_layers=2):
        super(GRUCell, self).__init__()
        self.lstm = nn.GRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(4, hidden_size)
        self.i2o = nn.Linear(4, 2)  
        
    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        out= self.check_forward_input(input)
        out = self.i2h(out)
        out = self.i2o(out)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            cx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        output = []
        for i in range(input.size(0)):
            hx, cx = nn.rnn(input[i], (hx, cx))
            output.append(hx)
        return output
    
class LSTMCell(nn.Module):
    
    def __init__(self, input_size=20, hidden_size=2, num_layers=2):
        super(LSTMCell, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(4, hidden_size)
        self.i2o = nn.Linear(4, 2)  
        
    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        out= self.check_forward_input(input)
        out = self.i2h(out)
        out = self.i2o(out)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            cx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        output = []
        for i in range(input.size(0)):
            hx, cx = nn.rnn(input[i], (hx, cx))
            output.append(hx)
        return output           

# 2.3 k-hot encoding of target values (0.5 points)

class LSTMCell_encoding(nn.Module):
    
    def __init__(self, input_size=20, hidden_size=2, num_layers=2):
        super(LSTMCell, self).__init__()
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(4, hidden_size)
        self.i2o = nn.Linear(4, 2)  
        
    def forward(self, input, hx=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        out= self.check_forward_input(input)
        out = self.i2h(out)
        out = self.i2o(out)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            cx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        output = []
        for i in range(input.size(0)):
            hx, cx = nn.rnn(input[i], (hx, cx))
            output.append(hx)
        return output

class loss(nn.Module):
    
    def __init__(self, input_size=20, hidden_size=2):
        super(loss, self).__init__()
        self.output = torch.full([input_size, hidden_size], 0.999)
        self.pos_weight = torch.ones([hidden_size])  # All weights are equal to 1
    def forward(self, input):    
        criterion = torch.nn.BCEWithLogitsLoss(self)
        return criterion
     
    
    
    
    
    