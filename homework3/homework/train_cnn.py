import torch
from torch.utils.data import DataLoader
from .models import  model_factory, save_model,CNNClassifier, save_model
from .utils import  load_data,ConfusionMatrix, load_data, LABEL_NAMES
import torchvision
import sys
import torch.utils.tensorboard as tb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train(args):
    from os import path
    model = CNNClassifier()
    model=model.cuda();

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    #Create the model ,loss function and optimizer
    lossFunction = torch.nn.CrossEntropyLoss()
    #accuracy= lambda o, l:((o>0).long()==l.long()).float()

    optimiz = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimiz,step_size=30,gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiz,[20,30],gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiz,'max')

    

    #load data for valid and train

    
    tl_load=load_data("data/train",batch_size=15)
   
    for epoch in range(50):
        # running_loss=0.0
        model.train()
        accuracies=[]
        for img, label in  tl_load:

            output=model.forward(img)
            loss=lossFunction(output.cuda(),label.cuda())
            #accuracies.extend(accuracy(output.cuda(),label.cuda()).detach.cpu().numpy())
            optimiz.zero_grad()
            # running_loss += loss.item()
            loss.backward()
            optimiz.step()
        #scheduler.step(np.mean(accuracies))
        scheduler.step()

    save_model(model)
    model.eval()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
