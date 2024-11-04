import torch
from spikingjelly.activation_based import neuron, surrogate, functional, layer


class RecordHook:
    def __init__(self, to_cpu=False):
        self.acs = 0
        self.firing_rate = []

    def __call__(self, module, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        if len(input) == 1:
            input = input[0]
        self.firing_rate.append(input.mean(axis=0).mean())
        if isinstance(module, torch.nn.Conv1d):
            print("Conv1d: ", input.shape, output.shape)
            self.acs += output.shape[1] * output.shape[2] * module.kernel_size[0] * input.shape[1] * self.firing_rate[
                -1]
        elif isinstance(module, torch.nn.Linear):
            print("Linear: ", input.shape, output.shape)
            self.acs += output.shape[1] * input.shape[1] * self.firing_rate[-1]

    def clear(self):
        self.acs = 0
        self.firing_rate = []

    def get_acs(self):
        return self.acs

    def get_firing_rate(self):
        return self.firing_rate


def get_acs(snn, data, T):
    hookers = []
    handlers = []
    names = []
    for name, layer in snn.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d)):
            names.append(name)
            hookers.append(RecordHook())
            handlers.append(layer.register_forward_hook(hookers[-1]))
    with torch.inference_mode():
        count = 0
        for t in range(T):
            count += snn(data[t])
    functional.reset_net(snn)
    for hanler in handlers:
        hanler.remove()
    return {name: hook.get_acs() for name, hook in zip(names, hookers)}
