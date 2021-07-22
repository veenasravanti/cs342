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


    """
    Your code here, modify your HW4 code
    
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    #if args.continue_training:
    #    model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_data('drive_data', num_workers=4, transform=transform)

    #det_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    size_loss = torch.nn.MSELoss(reduction='none')

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()

        for img, value in train_data:
            img, value = img.to(device), value.to(device)

            size_w, _ = value.max(dim=1, keepdim=True)

            det = model(img)
            # Continuous version of focal loss
            #p_det = torch.sigmoid(det * (1-2*gt_det))
            #det_loss_val = (det_loss(det, gt_det)*p_det).mean() / p_det.mean()
           # print('det',det.size())
           # print('value',value.size())
            loss_val = size_loss(det, value).mean()
            #loss_val = det_loss_val + size_loss_val * args.size_weight
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
    parser.add_argument('-n', '--num_epoch', type=int, default=25)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
