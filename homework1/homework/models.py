import torch
import torch.nn.functional as F
import csv

import torchvision
from PIL import Image
from torchvision import transforms


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        raise NotImplementedError('ClassificationLoss.forward')


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        dataset_path="C:\\Veena\\UT_MSDS\\395_DL\\hw1\\hw1\\homework1\\"
        with open(dataset_path+'data\\valid\\labels.csv') as f:
            data = csv.reader(f)
            next(data)
            self.data = list(data)
            for x in self.data:
                pic=Image.Open(dataset_path+"\\"+x[0])
                tensor = torchvision.transforms.ToTensor()(pic)
        """
        print("From Linear Classification")
        self.input_size=3*64*64
        self.linear1=torch.nn.Linear(self.input_size,1)
        self.activation=torch.nn.ReLu()


    def forward(self, x):
        """
        Your forward function receives a (B,3,64,64) tensor as an input and should return a (B,6) torch.Tensor (one value per class).
        """

        y=linear1(x)

        return self.linear1(self.activation(self.linear1(x.size(0),-1)))


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        raise NotImplementedError('MLPClassifier.__init__')

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        raise NotImplementedError('MLPClassifier.forward')


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
