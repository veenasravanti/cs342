import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
import torchvision
import sys
import torch.utils.tensorboard as tb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    from os import path
    model = FCN()
    model=model.cuda();
    
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    lossFunction = torch.nn.CrossEntropyLoss()
    cm = ConfusionMatrix();
    optimiz = torch.optim.Adam(model.parameters())
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiz,'max')
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiz,'max')

    tl_load = load_dense_data("dense_data/train",batch_size=15, transform = dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(), dense_transforms.ColorJitter(), dense_transforms.ToTensor()]))
    #tl_load=load_dense_data("dense_data/train",batch_size=15)
    valid_load=load_dense_data("dense_data/valid",batch_size=15,transform = dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(), dense_transforms.ColorJitter(), dense_transforms.ToTensor()]))

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    for epoch in range(50):
        
      model.train()
      
      for img, label in  tl_load:
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
      
      ioc_accuracy=[]
      for img, label in  valid_load:
             img=img.to(device)
             label=label.to(device)
             output=model.forward(img)
             cm.add(output.argmax(1),label)
              
      ioc_accuracy=cm.global_accuracy
      ioc_acc1=cm.iou
      print("epoch:",epoch,":",ioc_accuracy,"iou:",ioc_acc1)
      scheduler1.step(ioc_accuracy)
      scheduler2.step(ioc_acc1)
       
      if ioc_accuracy>.78 and ioc_acc1>.55:
        save_model(model,str(ioc_accuracy))
        #print("epoch", epoch ,"accuracies",acc)

      
    save_model(model,'')
    model.eval()

def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
