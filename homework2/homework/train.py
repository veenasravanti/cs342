from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb
import os


def train(args):
    from os import path
    model = CNNClassifier()
    lossFunction = torch.nn.CrossEntropyLoss()
    optimiz = torch.optim.SGD(model.parameters(), lr=0.01)
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    tl_load=load_data("data/train",batch_size=15)
   
    for epoch in range(30):
        # running_loss=0.0
        for img, label in  tl_load:

          output=model.forward(img)
          #print(output)
          #print(label)
          loss= lossFunction(output,label)
          #print(loss)
          optimiz.zero_grad()
            # running_loss += loss.item()
          loss.backward()
          optimiz.step()


    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
