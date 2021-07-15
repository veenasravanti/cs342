import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    from os import path
    model = Detector()
    model=model.cuda();
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    weights = [1/0.02929112, 1/0.0044619, 1/0.00411153]
    positive_weights = torch.Tensor(weights)
    lossFunction = torch.nn.BCEWithLogitsLoss(pos_weight = positive_weights[None, :, None, None] ).to(device)
   
    
    optimiz = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiz,'min')
    
    #should use both dense_transforms.ToTensor() and dense_transforms.ToHeatmap()
    tl_load = load_detection_data("dense_data/train",batch_size=15,transform =  dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(), dense_transforms.ColorJitter(contrast=0.5, brightness=0.5, saturation=0.5, hue=0.5),dense_transforms.ToTensor(), dense_transforms.ToHeatmap()]))                             
    valid_load = load_detection_data("dense_data/valid",batch_size=15,transform =  dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(), dense_transforms.ColorJitter(contrast=0.5, brightness=0.5, saturation=0.5, hue=0.5),dense_transforms.ToTensor(), dense_transforms.ToHeatmap()]))                             

    for epoch in range(50):
        
      model.train()
      
      for img,heatmap, size in  tl_load:
            
            img=img.to(device)
            heatmap=heatmap.to(device)
            output=model.forward(img)
            output=output.to(device)
            #label = label.type(torch.LongTensor).to(device)
            loss=lossFunction(output,heatmap)
            
            optimiz.zero_grad()
            # running_loss += loss.item()
            loss.backward()
            optimiz.step()

      print(f'Epoch {epoch}\t Training Loss: {loss/len(tl_load)}')
      

      for img,heatmap, size in  valid_load:
            
            img=img.to(device)
            heatmap=heatmap.to(device)
            output=model.forward(img)
            output=output.to(device)
                      
            optimiz.zero_grad()
            # running_loss += loss.item()
            optimiz.step()

      print("epoch", epoch)
      scheduler.step(loss)


            
    save_model(model)
    


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
