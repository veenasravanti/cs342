import torch
import torch.nn.functional as F
import csv
import sys

import torchvision
from PIL import Image
from torchvision import transforms


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        # inpSoftmax=torch.nn.functional.log_softmax(input)
        # output=torch.nn.functional.nll_loss(inpSoftmax,target)
        loss=torch.nn.CrossEntropyLoss()
        output=loss(input,target)
        return output


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_size=64*64*3
        output_size=6
        self.linearLayer=torch.nn.Linear(input_size,output_size)

    def forward(self, x):
        input=x.view(x.size(0),-1)
        return self.linearLayer(input)


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 64 * 64 * 3
        output_size = 6
        hidden_size=100
        self.linearLayer = torch.nn.Linear(input_size, hidden_size)
        self.linearLayer2= torch.nn.Linear(hidden_size, output_size)
        self.activationLayer=torch.nn.ReLU()




    def forward(self, x):

        input = x.view(x.size(0), -1)
        return self.linearLayer2(self.activationLayer(self.linearLayer(input)))

model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
