import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer



class SNNforauditory(nn.Module):
    def __init__(self, tau):
        super().__init__()


        self.fc = nn.Sequential(
            # nn.Flatten(),
            # layer.Dropout(0.7),
            nn.Linear(5, 32, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            # layer.Dropout(0.5),
            
            # # layer.Dropout(0.7),
            nn.Linear(32, 32, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            # layer.Dropout(0.5),
            #
            nn.Linear(32, 16, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            # layer.Dropout(0.2),

        )
        self.fc1 = nn.Sequential(
            # layer.Dropout(0.7),
            nn.Linear(16, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),

        )

    def forward(self, x: torch.Tensor):
        x=self.fc(x)
        x = self.fc1(x)
        return x
