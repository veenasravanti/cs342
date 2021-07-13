import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb


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
    lossFunction = torch.nn.CrossEntropyLoss()
    #cm = ConfusionMatrix();
    optimiz = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiz,'max')
    

    tl_load = load_detection_data("dense_data/train")
    valid_load=load_detection_data("dense_data/valid")

    for epoch in range(40):
        
      model.train()
      
      for img, label in  tl_load:
            print("came inside ")
            img=img.to(device)
            label=label.to(device)
            output=model.forward(img)
            output=output.to(device)
            label = label.type(torch.LongTensor).to(device)
            loss=lossFunction(output,label)
            
            optimiz.zero_grad()
            # running_loss += loss.item()
            loss.backward()
            optimiz.step()
      
      #scheduler.step()
      
      accuracy=[]
      for img, label in  valid_load:
             img=img.to(device)
             label=label.to(device)
             output=model.forward(img)
             #cm.add(output.argmax(1),label)
             accuracies.append(accuracy(output,label))
      acc= np.mean(accuracies)   
      
      print("epoch:",epoch,":",acc)
      scheduler.step(acc)
      
       
      if acc>.78 and acc>.55:
        save_model(model,str(acc))
        #print("epoch", epoch ,"accuracies",acc)
    
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
