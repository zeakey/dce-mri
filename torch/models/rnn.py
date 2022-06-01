import torch
from einops import rearrange


class RNNModel(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=6, batch_first=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.ktrans = torch.nn.Linear(num_layers * hidden_size, 1)
        self.kep = torch.nn.Linear(num_layers * hidden_size, 1)
        self.t0 = torch.nn.Linear(num_layers * hidden_size, 1)
    
    def forward(self, x):
        x = self.rnn(x)
        x = rearrange(x[1], 'l n h -> n (l h)', l=self.num_layers, h=self.hidden_size)
        ktrans = self.ktrans(x)
        kep = self.kep(x)
        t0 = self.t0(x)
        return ktrans, kep, t0

if __name__ == '__main__':
    model = RNNModel()
    x = torch.randn(32, 100, 1)
    print(model(x).shape)

        