import torch
from torch import nn


class FNN(nn.Module):
    def __init__(self, NL, NN):
        super(FNN, self).__init__()
        self.input_layer = nn.Linear(2, NN)
        self.hidden_layer = nn.ModuleList([nn.Linear(NN, NN) for i in range(NL)])
        self.output_layer = nn.Linear(NN, 1)

    def forward(self, x):
        out = self.act(self.input_layer(x))
        for layer in self.hidden_layer:
            out = self.act(layer(out))
        out = self.output_layer(out)
        return out

    def act(self, x):
        return torch.sigmoid(x)


