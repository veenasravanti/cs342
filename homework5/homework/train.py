from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    
    train_data = load_data('drive_data', num_workers=4, transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(), 
    dense_transforms.ColorJitter(contrast=0.5, brightness=0.5, saturation=0.5, hue=0.5),
    dense_transforms.ToTensor()]))

    size_loss = torch.nn.MSELoss(reduction='none')

    global_step = 0
    for epoch in range(25):
        model.train()

        for img, value in train_data:
            img, value = img.to(device), value.to(device)
            det = model(img)
            loss_val = size_loss(det, value).mean()
           
            if(global_step%20==0):
              print('global_step %-3d, loss_val %-3d' %(epoch, loss_val))
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if valid_logger is None or train_logger is None:
            print('epoch %-3d' %
                  (epoch))
        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    #parser.add_argument('-n', '--num_epoch', type=int, default=25)
   # parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    #parser.add_argument('-c', '--continue_training', action='store_true')
    #parser.add_argument('-t', '--transform',
                       # default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
