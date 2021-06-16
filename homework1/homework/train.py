import torch
from torch.utils.data import DataLoader

from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torchvision


def train(args):
    #Create the model ,loss function and optimizer
    model = model_factory[args.model]()
    lossFunction = ClassificationLoss()
    optimiz = torch.optim.SGD(model.parameters(), lr=0.01)

    #load data for valid and train

    #testloader = torch.utils.data.DataLoader(test_dl, batch_size=batch_size,
                                              #shuffle=True, num_workers=2)
    tl_load=load_data("data/train",batch_size=15)
    # trainloader = torch.utils.data.DataLoader(tl_load, batch_size=batch_size,
    #                                           shuffle=True,num_workers=2)

    # Run SGD     for several epochs
    for epoch in range(10):
        # running_loss=0.0
        for img, label in  tl_load:

            output=model.forward(img)
            loss=lossFunction(output,label)
            optimiz.zero_grad()
            # running_loss += loss.item()
            loss.backward()
            optimiz.step()

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
