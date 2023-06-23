import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.Tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        self.batch1 = nn.BatchNorm1d(n_outputs)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.Tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)
        self.batch2 = nn.BatchNorm1d(n_outputs)
        self.max_pool = nn.MaxPool1d(2)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.Tanh1, self.dropout1, self.batch1,
            self.conv2, self.chomp2, self.Tanh2, self.dropout2, self.batch2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.Tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.Tanh(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_features, num_layers, num_filters, kernel_sizes, dropout):
        super(TemporalConvNet, self).__init__()
        self.features = nn.ModuleList()
        self.num_features = num_features
        self.num_layers = num_layers
        self.num_filters = num_filters

        for signal in range(self.num_features):
            layers = []

            for i in range(self.num_layers):
                dilation_size = 2 ** i
                in_channel = 1 if i == 0 else self.num_filters
                out_channel = self.num_filters

                layers.append(
                    TemporalBlock(
                        in_channel, out_channel, kernel_sizes[signal], stride=1,
                        dilation=dilation_size, padding=(kernel_sizes[signal] - 1) * dilation_size,
                        dropout=dropout
                    )
                )

            self.features.append(nn.Sequential(*layers))

    def forward(self, x):
        inputs = []
        
        for i in range(x.size(2)):
            inputs.append(x[:, i, :].view(-1, 1, x.size(2)))
            inputs[i] = self.features[i](inputs[i])

        return inputs


class TCN(nn.Module):
    def __init__(self, **hypers):
        super(TCN, self).__init__()
        self.num_features = hypers.get('num_features', 1)
        self.num_layers = hypers.get('num_layers', 1)
        self.hidden_dim = hypers.get('hidden_dim', 16)
        self.kernel_sizes = hypers.get('kernel_sizes', [3] * self.num_features)
        self.dropout = hypers.get('dropout', 0.)
        self.output_size = hypers.get('output_size', 2)

        self.tcn = TemporalConvNet(self.num_features, 
                                   self.num_layers, 
                                   self.hidden_dim, 
                                   self.kernel_sizes, 
                                   dropout=self.dropout
                                   )
        
        self.fc = nn.Sequential(nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.hidden_dim * self.num_features, self.output_size)
                                )
    
    def forward(self, x):
        x = self.tcn(x)
        processed_features = []

        for i in range(len(x)):
            processed_features.append(x[i][:, :, -1].view(x[i].size(0), -1))

        x = torch.cat(processed_features, dim=1)
        x = self.fc(x)

        return x
