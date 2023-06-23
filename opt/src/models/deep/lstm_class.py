import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, output_size=2, **hypers):
        super().__init__()
        self.input_dim = hypers.get('input_dim', 1)
        self.hidden_dim = hypers.get('hidden_dim', 1)
        self.num_layers = hypers.get('num_layers', 1)
        self.dropout = hypers.get('dropout', 0.)
        self.output_size = output_size
        self.bidirectional = hypers.get('bidirectional', False)
        self.device = hypers.get('device', 'cuda')
        
        self.lstm = nn.LSTM(self.input_dim, 
                            self.hidden_dim, 
                            self.num_layers, 
                            batch_first=True, 
                            dropout=self.dropout, 
                            bidirectional=self.bidirectional)
        
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        
    def forward(self, x):
        x = x.transpose(1, 2)

        if self.bidirectional:
          h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
          c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        _, (hidden, _) = self.lstm(x,(h0.detach(), c0.detach()))
        out = self.fc(hidden[-1])
        return out