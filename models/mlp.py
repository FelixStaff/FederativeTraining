import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128, bias=bias),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10, bias=bias)
        )

    def forward(self, x):
        # Apply the softmax activation function to the output layer
        x = self.net(x)
        return x
